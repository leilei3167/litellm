"""
Regression tests for the openai-like / databricks streaming chunk parser.

OpenAI-compatible servers (e.g. Vertex AI Model Garden vLLM endpoints, Alibaba
compatible-mode DeepSeek) stream reasoning on ``delta.reasoning_content`` and
may send a final usage-only chunk with an empty ``choices`` list when
``stream_options.include_usage`` is set. The parser must preserve both.
"""

from litellm.llms.databricks.streaming_utils import ModelResponseIterator
from litellm.types.utils import ModelResponseStream


def test_chunk_parser_handles_empty_choices_usage_chunk():
    """A usage-only final chunk (empty choices) must not raise IndexError."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    usage_only_chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "total_tokens": 28,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }

    result = iterator.chunk_parser(chunk=usage_only_chunk)

    assert isinstance(result, ModelResponseStream)
    assert result.choices == []
    assert result.usage is not None
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 8
    assert result.usage.completion_tokens_details is not None
    assert result.usage.completion_tokens_details.reasoning_tokens == 5


def test_chunk_parser_empty_choices_without_usage():
    """An empty-choices chunk with no usage block returns usage=None, no error."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert isinstance(result, ModelResponseStream)
    assert result.choices == []
    assert getattr(result, "usage", None) is None


def test_chunk_parser_normal_content_chunk_still_works():
    """A regular content chunk is unaffected by the empty-choices guard."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert isinstance(result, ModelResponseStream)
    assert result.choices[0].delta.content == "hi"


def test_chunk_parser_preserves_reasoning_content_delta():
    """Reasoning-only deltas must keep delta.reasoning_content (issue #41049)."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek-v4.1-flash",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Let me work through this step by step.",
                },
                "finish_reason": None,
            }
        ],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert isinstance(result, ModelResponseStream)
    assert result.choices[0].delta.content is None
    assert result.choices[0].delta.reasoning_content == "Let me work through this step by step."


def test_chunk_parser_maps_reasoning_field_to_reasoning_content():
    """Some OpenAI-compatible providers stream delta.reasoning instead."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {"content": None, "reasoning": "thinking aloud"},
                "finish_reason": None,
            }
        ],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert result.choices[0].delta.reasoning_content == "thinking aloud"


def test_openai_like_stream_wrapper_keeps_reasoning_and_usage():
    """CustomStreamWrapper must not drop reasoning-only openai_like chunks."""
    from unittest.mock import MagicMock

    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    logging_obj = MagicMock()
    logging_obj.model_call_details = {
        "litellm_params": {},
        "custom_llm_provider": "openai_like",
    }
    logging_obj.call_type = "completion"
    logging_obj.messages = []

    wrapper = CustomStreamWrapper(
        completion_stream=None,
        model="deepseek-v4.1-flash",
        custom_llm_provider="openai_like",
        logging_obj=logging_obj,
        stream_options={"include_usage": True},
    )

    reasoning_chunk = iterator.chunk_parser(
        {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-v4.1-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "step by step",
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    reasoning_out = wrapper.chunk_creator(reasoning_chunk)
    assert reasoning_out is not None
    assert reasoning_out.choices[0].delta.reasoning_content == "step by step"

    usage_chunk = iterator.chunk_parser(
        {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-v4.1-flash",
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 80,
                "total_tokens": 92,
                "completion_tokens_details": {"reasoning_tokens": 48},
            },
        }
    )
    usage_out = wrapper.chunk_creator(usage_chunk)
    assert usage_out is not None
    assert usage_out.usage is not None
    assert usage_out.usage.completion_tokens_details is not None
    assert usage_out.usage.completion_tokens_details.reasoning_tokens == 48
