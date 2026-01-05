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

    def _sanitize_schema(self, schema: Any) -> dict[str, Any] | Any:
        """Reduce schemas to Gemini's supported subset."""
        if not isinstance(schema, dict):
            return schema

        working = dict(schema)
        working.pop("$defs", None)

        if "anyOf" in working:
            variants = working.get("anyOf") or []
            non_null = [
                variant
                for variant in variants
                if not (isinstance(variant, dict) and variant.get("type") == "null")
            ]
            chosen = non_null[0] if non_null else (variants[0] if variants else {})
            working = chosen if isinstance(chosen, dict) else {}

        working.pop("$ref", None)

        allowed_keys = {"type", "properties", "items", "required", "enum", "description"}
        sanitized: dict[str, Any] = {
            key: value for key, value in working.items() if key in allowed_keys
        }

        if properties := sanitized.get("properties"):
            if isinstance(properties, dict):
                sanitized["properties"] = {
                    name: self._sanitize_schema(prop_schema)
                    for name, prop_schema in properties.items()
                    if isinstance(prop_schema, (dict, list))
                }
            sanitized.setdefault("type", "object")

        if "items" in sanitized and isinstance(sanitized["items"], (dict, list)):
            sanitized["items"] = self._sanitize_schema(sanitized["items"])

        return sanitized

    def _tool_definitions(self, tools: list[AvailableTool] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        # Gemini expects a single tools entry containing all declarations.
        function_declarations = [
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": self._sanitize_schema(tool.function.parameters),
            }
            for tool in tools
        ]
        return [{"functionDeclarations": function_declarations}]

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
                            emit_llm_log(streaming=True, request="", response=f"tc send : {tc.thought_signature}")
                            args = tc.function.arguments or "{}"
                            try:
                                parsed_args = json.loads(args)
                            except json.JSONDecodeError:
                                parsed_args = {"_raw": args}
                            function_call_payload = {
                                "name": tc.function.name or "",
                                "args": parsed_args,
                                **({"id": tc.id} if tc.id else {}),
                            }
                            parts.append(
                                {
                                    "functionCall": function_call_payload,
                                    **(
                                        {"thoughtSignature": tc.thought_signature}
                                        if tc.thought_signature
                                        else {}
                                    ),
                                }
                            )
                            # Preserve index ordering hint if provided
                            #if tc.index is not None and parts[-1]["functionCall"].get("index") is None:
                            #    parts[-1]["functionCall"]["index"] = tc.index
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

        return contents, system_instruction

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
                signature = part.get("thoughtSignature")
                args_obj = fn_call.get("args", {})
                args = json.dumps(args_obj) if not isinstance(args_obj, str) else args_obj
                tool_call = ToolCall(
                    id=fn_call.get("id"),
                    index=fn_call.get("index", idx),
                    function=FunctionCall(
                        name=fn_call.get("name"),
                        arguments=args,
                    ),
                )
                if signature:
                    emit_llm_log(streaming=True, request="", response=f"thoughtSignature: {signature}")
                    tool_call.thought_signature = signature
                tool_calls.append(tool_call)
        return "".join(text_parts), tool_calls

    def _parse_candidate(self, data: dict[str, Any]) -> LLMChunk:
        candidates = data.get("candidates") or []
        candidate = candidates[0] or {} if candidates else {}
        content = candidate.get("content") or {}
        parts = content.get("parts") or []

        message_text, tool_calls = self._parse_parts(parts)
        usage_meta = data.get("usageMetadata") or candidate.get("usageMetadata") or {}
        prompt_tokens = usage_meta.get("promptTokenCount", 0)

        completion_tokens = (
            usage_meta.get("candidatesTokenCount", 0)
        )
        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
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
        emit_llm_log(streaming=False, request=body, response=json.dumps(response_data, ensure_ascii=False))
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
        async with self._get_client().stream(
            "POST", url, json=body, headers=headers, params=params
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
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
        emit_llm_log(
            streaming=True, request=body, response="\n".join(stream_chunks)
        )

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

