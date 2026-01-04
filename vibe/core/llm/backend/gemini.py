from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from copy import deepcopy
import json
import os
from typing import TYPE_CHECKING, Any

import httpx

from vibe.core.llm.backend.generic import emit_llm_log
from vibe.core.llm.exceptions import BackendErrorBuilder
from vibe.core.types import (
    AvailableTool,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StrToolChoice,
    ToolCall,
)
from vibe.core.utils import async_generator_retry, async_retry

if TYPE_CHECKING:
    from vibe.core.config import ModelConfig, ProviderConfig

if os.getenv("VIBE_DEBUG_GEMINI"):
    # Lightweight optional logging without forcing logger import on hot paths
    import logging

    _logger = logging.getLogger(__name__)
else:
    _logger = None


class GeminiThoughtSignatureTracker:
    """Preserve and re-attach Gemini 3 thought signatures across tool calls.

    Gemini 3 requires the ``thoughtSignature`` attached to the first
    ``functionCall`` part of every step to be echoed back in subsequent requests.
    Dropping the signature causes a 4xx validation error. This helper records the
    signature from a model response and can inject it back into the conversation
    history before the next request is sent.
    """

    def __init__(self, *, prefer_snake_case: bool = False) -> None:
        self._signatures: list[str] = []
        self._prefer_snake_case = prefer_snake_case

    def record_model_response(
        self,
        *,
        parts: Sequence[Mapping[str, Any]] | None = None,
        message: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Capture the first thought signature from a model message.

        Args:
            parts: The ``parts`` array from a Gemini model response.
            message: Full model message that contains ``parts``.

        Returns:
            The captured signature or ``None`` when no signature was found.
        """
        resolved_parts = self._resolve_parts(parts, message)
        if not self._contains_function_call(resolved_parts):
            return None

        signature = self._find_first_signature(resolved_parts)
        if signature:
            self._signatures.append(signature)
        return signature

    def seed_from_history(self, contents: Sequence[Mapping[str, Any]]) -> None:
        """Populate the tracker from an existing conversation history."""
        for message in contents:
            if message.get("role") != "model":
                continue
            self.record_model_response(parts=message.get("parts") or [])

    def ensure_signatures(
        self,
        contents: Sequence[Mapping[str, Any]],
        *,
        prefer_snake_case: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Return a copy of ``contents`` with required thought signatures attached."""
        updated: list[dict[str, Any]] = []
        signature_index = 0

        for message in contents:
            as_dict = dict(message)
            role = as_dict.get("role")
            parts = as_dict.get("parts") or []
            if role != "model" or not self._contains_function_call(parts):
                updated.append(deepcopy(as_dict))
                continue

            expected = (
                self._signatures[signature_index]
                if signature_index < len(self._signatures)
                else None
            )
            signature_index += 1

            updated_parts = self._attach_signature(
                parts,
                expected_signature=expected,
                prefer_snake_case=prefer_snake_case,
            )
            updated.append({"role": role, "parts": updated_parts})

        # Trim stale signatures when history is compacted or truncated
        if signature_index < len(self._signatures):
            self._signatures = self._signatures[:signature_index]

        return updated

    def reset(self) -> None:
        self._signatures.clear()

    @property
    def signatures(self) -> list[str]:
        """Return a copy of all captured signatures."""
        return list(self._signatures)

    def _resolve_parts(
        self,
        parts: Sequence[Mapping[str, Any]] | None,
        message: Mapping[str, Any] | None,
    ) -> list[Mapping[str, Any]]:
        if parts is not None:
            return list(parts)
        if message is None:
            raise ValueError("Either 'parts' or 'message' must be provided.")
        resolved_parts = message.get("parts")
        if resolved_parts is None:
            raise ValueError("Gemini model message is missing the 'parts' field.")
        return list(resolved_parts)

    def _find_first_signature(
        self, parts: Sequence[Mapping[str, Any]]
    ) -> str | None:
        for part in parts:
            if not self._is_function_call(part):
                continue
            if signature := self._extract_signature(part):
                return signature
        return None

    def _contains_function_call(self, parts: Sequence[Mapping[str, Any]]) -> bool:
        return any(self._is_function_call(part) for part in parts)

    def _attach_signature(
        self,
        parts: Sequence[Mapping[str, Any]],
        *,
        expected_signature: str | None,
        prefer_snake_case: bool | None,
    ) -> list[dict[str, Any]]:
        updated_parts: list[dict[str, Any]] = []
        signature_applied = False

        for part in parts:
            part_copy: dict[str, Any] = dict(part)
            if not self._is_function_call(part_copy):
                updated_parts.append(deepcopy(part_copy))
                continue

            if signature_applied:
                updated_parts.append(deepcopy(part_copy))
                continue

            signature_applied = True
            existing_signature = self._extract_signature(part_copy)
            if existing_signature:
                updated_parts.append(deepcopy(part_copy))
                continue

            if expected_signature is None:
                raise ValueError(
                    "Missing thought signature for Gemini function call step. "
                    "Call 'record_model_response' with the previous model output "
                    "before building the next request."
                )

            signature_key = self._signature_key(part_copy, prefer_snake_case)
            part_copy[signature_key] = expected_signature
            updated_parts.append(part_copy)

        return updated_parts

    @staticmethod
    def _is_function_call(part: Mapping[str, Any]) -> bool:
        return "functionCall" in part or "function_call" in part

    @staticmethod
    def _extract_signature(part: Mapping[str, Any]) -> str | None:
        match part:
            case {"thoughtSignature": str(signature)}:
                return signature
            case {"thought_signature": str(signature)}:
                return signature
        return None

    def _signature_key(
        self, part: Mapping[str, Any], prefer_snake_case: bool | None
    ) -> str:
        if "thoughtSignature" in part:
            return "thoughtSignature"
        if "thought_signature" in part:
            return "thought_signature"

        if prefer_snake_case is None:
            prefer_snake_case = self._prefer_snake_case
        return "thought_signature" if prefer_snake_case else "thoughtSignature"


class GeminiBackend:
    """Native Gemini 3 backend with thought signature handling."""

    def __init__(self, *, provider: "ProviderConfig", timeout: float = 720.0) -> None:
        self._provider = provider
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._api_key = (
            os.getenv(self._provider.api_key_env_var)
            if self._provider.api_key_env_var
            else None
        )
        self._tracker = GeminiThoughtSignatureTracker()

    async def __aenter__(self) -> "GeminiBackend":
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                http2=True,
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise ValueError("Client not initialized")
        return self._client

    def _generation_url(self, model_name: str, *, stream: bool = False) -> str:
        suffix = ":streamGenerateContent" if stream else ":generateContent"
        return f"{self._provider.api_base}/models/{model_name}{suffix}"

    def _count_tokens_url(self, model_name: str) -> str:
        return f"{self._provider.api_base}/models/{model_name}:countTokens"

    def _headers(self, extra_headers: dict[str, str] | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-Goog-Api-Key"] = self._api_key
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _tool_definitions(self, tools: list[AvailableTool] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "functionDeclarations": [
                    {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                    }
                ]
            }
            for tool in tools
        ]

    def _tool_config(
        self, tool_choice: StrToolChoice | AvailableTool | None
    ) -> dict[str, Any] | None:
        if tool_choice is None or tool_choice == "auto":
            return None

        mode = "AUTO"
        allowed: list[str] | None = None

        match tool_choice:
            case "none":
                mode = "NONE"
            case "any" | "required":
                mode = "ANY"
            case AvailableTool():
                mode = "ANY"
                allowed = [tool_choice.function.name]

        config: dict[str, Any] = {"functionCallingConfig": {"mode": mode}}
        if allowed:
            config["functionCallingConfig"]["allowedFunctionNames"] = allowed
        return config

    def _build_system_instruction(self, messages: list[LLMMessage]) -> tuple[dict[str, Any] | None, list[LLMMessage]]:
        if not messages or messages[0].role != Role.system:
            return None, messages
        system_msg, *rest = messages
        return (
            {"parts": [{"text": system_msg.content or ""}]},
            rest,
        )

    def _convert_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        system_instruction, remaining = self._build_system_instruction(messages)
        contents: list[dict[str, Any]] = []

        for msg in remaining:
            match msg.role:
                case Role.user:
                    contents.append(
                        {
                            "role": "user",
                            "parts": [{"text": msg.content or ""}],
                        }
                    )
                case Role.assistant:
                    parts: list[dict[str, Any]] = []
                    if msg.tool_calls:
                        for idx, tc in enumerate(sorted(msg.tool_calls, key=lambda t: t.index or 0)):
                            args = tc.function.arguments or "{}"
                            try:
                                parsed_args = json.loads(args)
                            except json.JSONDecodeError:
                                parsed_args = {"_raw": args}
                            parts.append(
                                {
                                    "functionCall": {
                                        "name": tc.function.name or "",
                                        "args": parsed_args,
                                        **({"id": tc.id} if tc.id else {}),
                                    }
                                }
                            )
                            # Preserve index ordering hint if provided
                            if tc.index is not None and parts[-1]["functionCall"].get("index") is None:
                                parts[-1]["functionCall"]["index"] = tc.index
                    if msg.content:
                        parts.append({"text": msg.content})
                    contents.append({"role": "model", "parts": parts or [{"text": ""}]})
                case Role.tool:
                    payload: Any = msg.content
                    if isinstance(msg.content, str):
                        try:
                            payload = json.loads(msg.content)
                        except json.JSONDecodeError:
                            payload = {"output": msg.content}
                    contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "functionResponse": {
                                        "name": msg.name or "",
                                        "response": payload,
                                        **({"id": msg.tool_call_id} if msg.tool_call_id else {}),
                                    }
                                }
                            ],
                        }
                    )
                case Role.system:
                    # Secondary system messages are treated as user text to avoid loss
                    contents.append(
                        {
                            "role": "user",
                            "parts": [{"text": msg.content or ""}],
                        }
                    )

        contents_with_signatures = self._tracker.ensure_signatures(contents)
        return contents_with_signatures, system_instruction

    def _generation_body(
        self,
        *,
        model: "ModelConfig",
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
    ) -> dict[str, Any]:
        contents, system_instruction = self._convert_messages(messages)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                **({"maxOutputTokens": max_tokens} if max_tokens is not None else {}),
            },
        }

        if system_instruction:
            body["systemInstruction"] = system_instruction

        tool_defs = self._tool_definitions(tools)
        if tool_defs:
            body["tools"] = tool_defs
        if tool_config := self._tool_config(tool_choice):
            body["toolConfig"] = tool_config

        return body

    def _parse_parts(self, parts: list[dict[str, Any]]) -> tuple[str, list[ToolCall]]:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for idx, part in enumerate(parts):
            if "text" in part and isinstance(part["text"], str):
                text_parts.append(part["text"])
            if "functionCall" in part:
                fn_call = part["functionCall"] or {}
                args_obj = fn_call.get("args", {})
                args = json.dumps(args_obj) if not isinstance(args_obj, str) else args_obj
                tool_calls.append(
                    ToolCall(
                        id=fn_call.get("id"),
                        index=fn_call.get("index", idx),
                        function=FunctionCall(
                            name=fn_call.get("name"),
                            arguments=args,
                        ),
                    )
                )
        return "".join(text_parts), tool_calls

    def _parse_candidate(self, data: dict[str, Any]) -> LLMChunk:
        candidates = data.get("candidates") or []
        if not candidates:
            return LLMChunk(
                message=LLMMessage(role=Role.assistant, content=""),
                usage=LLMUsage(prompt_tokens=0, completion_tokens=0),
            )

        candidate = candidates[0] or {}
        content = candidate.get("content") or {}
        parts = content.get("parts") or []

        self._tracker.record_model_response(parts=parts)

        message_text, tool_calls = self._parse_parts(parts)
        usage_meta = data.get("usageMetadata") or candidate.get("usageMetadata") or {}
        usage = LLMUsage(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
        )

        return LLMChunk(
            message=LLMMessage(
                role=Role.assistant,
                content=message_text,
                tool_calls=tool_calls or None,
            ),
            usage=usage,
        )

    async def _post_json(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        request_payload = json.dumps(body, ensure_ascii=False)
        response = await self._get_client().post(url, json=body, headers=headers, params=params)
        response.raise_for_status()
        response_data = response.json()
        emit_llm_log(streaming=False, request=request_payload, response=json.dumps(response_data, ensure_ascii=False))
        return response_data

    async def _post_json_stream(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        params: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_payload = json.dumps(body, ensure_ascii=False)
        stream_chunks: list[str] = []
        print(url, request_payload, headers, params)
        async with self._get_client().stream(
            "POST", url, json=body, headers=headers, params=params
        ) as response:
            response.raise_for_status()
            print(response.status_code, response.headers)
            async for line in response.aiter_lines():
                print(line)
                if not line:
                    continue
                payload_str = line.removeprefix("data:").strip()
                if not payload_str or payload_str == "[DONE]":
                    continue
                stream_chunks.append(payload_str)
                try:
                    yield json.loads(payload_str)
                except json.JSONDecodeError:
                    if _logger:
                        _logger.debug("Skipping non-JSON stream chunk: %s", payload_str)
        emit_llm_log(streaming=True, request=request_payload, response="\n".join(stream_chunks))

    async def complete(
        self,
        *,
        model: "ModelConfig",
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
    ) -> LLMChunk:
        headers = self._headers(extra_headers)
        params: dict[str, Any] = {}
        body = self._generation_body(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )
        url = self._generation_url(model.name, stream=False)

        try:
            data = await self._post_json(url, body, headers, params)
            return self._parse_candidate(data)
        except httpx.HTTPStatusError as e:
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                response=e.response,
                headers=e.response.headers,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    async def complete_streaming(
        self,
        *,
        model: "ModelConfig",
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
    ) -> AsyncGenerator[LLMChunk, None]:
        headers = self._headers(extra_headers)
        params: dict[str, Any] = {"alt": "sse"}
        body = self._generation_body(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )
        url = self._generation_url(model.name, stream=True)

        try:
            async for payload in self._post_json_stream(url, body, headers, params):
                yield self._parse_candidate(payload)
        except httpx.HTTPStatusError as e:
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                response=e.response,
                headers=e.response.headers,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    async def count_tokens(
        self,
        *,
        model: "ModelConfig",
        messages: list[LLMMessage],
        temperature: float = 0.0,
        tools: list[AvailableTool] | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> int:
        headers = self._headers(extra_headers)
        params: dict[str, Any] = {}
        contents, system_instruction = self._convert_messages(messages)
        body: dict[str, Any] = {"contents": contents}
        if system_instruction:
            body["systemInstruction"] = system_instruction
        url = self._count_tokens_url(model.name)

        try:
            data = await self._post_json(url, body, headers, params)
            return int(data.get("totalTokens", 0))
        except httpx.HTTPStatusError as e:
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                response=e.response,
                headers=e.response.headers,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

