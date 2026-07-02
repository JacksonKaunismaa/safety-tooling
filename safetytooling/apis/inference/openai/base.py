import asyncio
import logging
import os
import time
from pathlib import Path
from traceback import format_exc
from typing import Optional

import openai

from safetytooling.data_models import LLMResponse

from ..model import InferenceAPIModel
from .utils import COMPLETION_MODELS

# Per-call hard timeout. Defaults high (patient clients cost nothing; timeouts cascade):
# against a self-hosted vLLM fleet, a request can legitimately queue + generate for >20 min
# under load, and timing it out abandons work the server keeps computing, then retries it
# from scratch — a self-sustaining throughput collapse. Override via SAFETYTOOLING_API_TIMEOUT
# if a workload needs to fail faster (e.g. flaky public endpoints where an hour-long hang on a
# dead connection is worse than a retry).
API_CALL_TIMEOUT_SECONDS = float(os.environ.get("SAFETYTOOLING_API_TIMEOUT", 3600))

LOGGER = logging.getLogger(__name__)


class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    def __init__(self, refresh_rate, total=0, throughput=0):
        self.refresh_rate = refresh_rate
        self.total = total
        self.throughput = throughput
        self.last_update_time = time.time()
        self.start_time = time.time()
        self.value = self.refresh_rate

    def _replenish(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(
            self.refresh_rate,
            self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60,
        )
        self.last_update_time = curr_time
        self.throughput = self.total / (curr_time - self.start_time) * 60

    def geq(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(amount), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount
        self.total += amount


class OpenAIModel(InferenceAPIModel):
    def __init__(
        self,
        frac_rate_limit: float,
        prompt_history_dir: Path | None = None,
        base_url: str | None = None,
        openai_api_key: str | None = None,
    ):
        self.frac_rate_limit = frac_rate_limit
        self.prompt_history_dir = prompt_history_dir
        self.model_ids = set()
        self.base_url = base_url
        self.openai_api_key = openai_api_key

        # timeout: the SDK's default is 600s with 2 silent internal retries — a request whose
        # generation legitimately takes >10 min (long CoT on a loaded self-hosted fleet) times
        # out and re-generates forever, so it must track API_CALL_TIMEOUT_SECONDS, not the
        # default. max_retries=0 because our outer retry loop (with backoff + logging) owns
        # retries; hidden SDK retries multiply long timeouts invisibly.
        client_kwargs: dict = {
            "base_url": self.base_url,
            "timeout": API_CALL_TIMEOUT_SECONDS,
            "max_retries": 0,
        }
        if openai_api_key:
            self.aclient = openai.AsyncClient(api_key=openai_api_key, **client_kwargs)
        elif "OPENAI_API_KEY" in os.environ:
            self.aclient = openai.AsyncClient(**client_kwargs)
        else:
            self.aclient = None

        self.token_capacity = dict()
        self.request_capacity = dict()
        self.lock_add = asyncio.Lock()
        self.lock_consume = asyncio.Lock()

    @staticmethod
    async def _get_dummy_response_header(model_id: str):
        raise NotImplementedError

    @staticmethod
    def _count_prompt_token_capacity(prompt, **kwargs) -> int:
        raise NotImplementedError

    async def _make_api_call(self, prompt, model_id, start_time, **params) -> list[LLMResponse]:
        raise NotImplementedError

    @staticmethod
    def _print_prompt_and_response(prompt, responses):
        raise NotImplementedError

    async def add_model_id(self, model_id: str):
        if self.aclient is None:
            raise RuntimeError("OpenAI API key must be set either via parameter or OPENAI_API_KEY environment variable")

        if model_id in self.model_ids:
            return

        self.model_ids.add(model_id)

        # make dummy request to get token and request capacity
        model_metadata = await self._get_dummy_response_header(model_id)
        token_capacity = int(model_metadata["x-ratelimit-limit-tokens"])
        request_capacity = int(model_metadata["x-ratelimit-limit-requests"])
        print(f"got capacities for model {model_id}: {token_capacity}, {request_capacity}")
        tokens_consumed = token_capacity - int(model_metadata["x-ratelimit-remaining-tokens"])
        requests_consumed = request_capacity - int(model_metadata["x-ratelimit-remaining-requests"])
        print(f"consumed capacities for model {model_id}: {tokens_consumed}, {requests_consumed}")
        token_cap = token_capacity * self.frac_rate_limit
        request_cap = request_capacity * self.frac_rate_limit
        if model_id in COMPLETION_MODELS:
            token_cap *= 10000  # openai does not track token limit so we can increase it

        print(f"setting cap for model {model_id}: {token_cap}, {request_cap}")
        token_capacity = Resource(token_cap)
        request_capacity = Resource(request_cap)
        token_capacity.consume(min(token_cap, tokens_consumed))
        request_capacity.consume(min(request_cap, requests_consumed))
        self.token_capacity[model_id] = token_capacity
        self.request_capacity[model_id] = request_capacity

    async def __call__(
        self,
        model_id: str,
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()

        async def attempt_api_call():
            async with self.lock_consume:
                request_capacity, token_capacity = (
                    self.request_capacity[model_id],
                    self.token_capacity[model_id],
                )
                if request_capacity.geq(1) and token_capacity.geq(token_count):
                    request_capacity.consume(1)
                    token_capacity.consume(token_count)
                else:
                    await asyncio.sleep(0.01)

            # Make the API call outside the lock
            return await asyncio.wait_for(
                self._make_api_call(prompt, model_id, start, **kwargs),
                timeout=API_CALL_TIMEOUT_SECONDS,
            )

        async with self.lock_add:
            await self.add_model_id(model_id)
        token_count = self._count_prompt_token_capacity(prompt, **kwargs)
        assert self.token_capacity[model_id].refresh_rate >= token_count, "Prompt is too long for any model to handle."
        responses: Optional[list[LLMResponse]] = None
        for i in range(max_attempts):
            try:
                responses = await attempt_api_call()
                if responses is not None and not all(is_valid(response.completion) for response in responses):
                    raise RuntimeError(f"Invalid responses according to is_valid {responses}")
            except (TypeError, openai.NotFoundError) as e:
                raise e
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break

        if responses is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        if print_prompt_and_response:
            prompt.pretty_print(responses)

        end = time.time()
        LOGGER.debug(f"Completed call to {model_id} in {end - start}s.")
        return responses
