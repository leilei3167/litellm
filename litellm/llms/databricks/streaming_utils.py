import json
from typing import Final

import litellm
from litellm import verbose_logger
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream


class ModelResponseIterator:
    def __init__(self, streaming_response, sync_stream: bool):
        self.streaming_response = streaming_response

    @staticmethod
    def _map_reasoning_to_reasoning_content(choices: list) -> list:
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta", {})
            if isinstance(delta, dict) and "reasoning" in delta:
                delta["reasoning_content"] = delta.pop("reasoning")
        return choices

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            error: Final = chunk.get("error")
            if error:
                raise ValueError(f"Error in stream chunk: {error}")

            choices: Final = self._map_reasoning_to_reasoning_content(list(chunk.get("choices") or []))
            kwargs: Final[dict] = {
                "id": chunk.get("id"),
                "object": "chat.completion.chunk",
                "created": chunk.get("created"),
                "model": chunk.get("model"),
                "choices": choices,
            }
            if "usage" in chunk and chunk["usage"] is not None:
                kwargs["usage"] = chunk["usage"]
            return ModelResponseStream(**kwargs)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        self.response_iterator = self.streaming_response
        return self

    def __next__(self):
        if not hasattr(self, "response_iterator"):
            self.response_iterator = self.streaming_response
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            chunk = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(chunk) or ""
            chunk = chunk.strip()
            if len(chunk) > 0:
                json_chunk: Final = json.loads(chunk)
                return self.chunk_parser(chunk=json_chunk)
            else:
                return GenericStreamingChunk(
                    text="",
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=0,
                    tool_use=None,
                )
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            verbose_logger.debug(
                "Error parsing chunk: %s,\nReceived chunk: %s. Defaulting to empty chunk here.", e, chunk
            )
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")
        except Exception as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            chunk = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(chunk) or ""
            chunk = chunk.strip()
            if chunk == "[DONE]":
                raise StopAsyncIteration
            if len(chunk) > 0:
                json_chunk: Final = json.loads(chunk)
                return self.chunk_parser(chunk=json_chunk)
            else:
                return GenericStreamingChunk(
                    text="",
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=0,
                    tool_use=None,
                )
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            verbose_logger.debug(
                "Error parsing chunk: %s,\nReceived chunk: %s. Defaulting to empty chunk here.", e, chunk
            )
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )
