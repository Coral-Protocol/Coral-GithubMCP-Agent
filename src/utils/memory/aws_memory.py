# utils/memory/aws_memory.py
import hashlib
from datetime import datetime
from typing import Dict, Any
from os import getenv
from bedrock_agentcore.memory import MemoryClient
import logging

logger = logging.getLogger(__name__)

EMAIL_FOR_ACTOR = "suman@coralprotocol.ai"
AWS_ACCESS_KEY_ID = getenv("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = getenv("AWS_SECRET_ACCESS_KEY", None)
AWS_DEFAULT_REGION = getenv("AWS_DEFAULT_REGION", None)
MEMORY_ID = getenv("MEMORY_ID", None)

MAX_MEMORY_EVENTS = 10
MAX_PAIRS = 10


def extract_conversation_history(events, max_pairs: int = MAX_PAIRS) -> Dict[str, Any]:
    pairs = []
    pending_user_msg = None

    for event in reversed(events):
        if len(pairs) >= max_pairs:
            break
        for item in event.get("payload", []):
            conv = item.get("conversational", {})
            role = conv.get("role")
            text = conv.get("content", {}).get("text", "").strip()
            if not text or role not in ("USER", "ASSISTANT"):
                continue
            if role == "USER":
                pending_user_msg = text
            else:
                if pending_user_msg:
                    pairs.append({"user": pending_user_msg, "assistant": text})
                    pending_user_msg = None

    if pending_user_msg:
        pairs.append({"user": pending_user_msg, "assistant": ""})

    return {"conversation": pairs}


async def get_aws_short_term_memory(coral_session_id: str) -> Dict[str, Any]:
    try:
        memory_client = MemoryClient(region_name=AWS_DEFAULT_REGION)
        actor_id = hashlib.md5(EMAIL_FOR_ACTOR.encode("utf-8")).hexdigest()

        events = memory_client.list_events(
            memory_id=MEMORY_ID,
            actor_id=actor_id,
            session_id=coral_session_id,
            max_results=MAX_MEMORY_EVENTS,
        )

        conversation = extract_conversation_history(events, MAX_PAIRS)
        logger.info(f"Extracted conversation: {conversation}")
        return conversation

    except Exception as e:
        logger.error(f"Failed to load AWS memory: {e}")
        return {"conversation": []}


async def upload_aws_short_term_memory(coral_session_id: str, user_message: str, assistant_response: str) -> None:
    try:
        client = MemoryClient(region_name=AWS_DEFAULT_REGION)
        actor_id = hashlib.md5(EMAIL_FOR_ACTOR.encode("utf-8")).hexdigest()

        messages = [
            (user_message.strip(), "USER"),
            (assistant_response.strip(), "ASSISTANT")
        ]

        client.create_event(
            memory_id=MEMORY_ID,
            actor_id=actor_id,
            session_id=coral_session_id,
            messages=messages,
        )

        logger.info("Successfully saved to AWS memory")

    except Exception as e:
        logger.error(f"Memory upload failed: {e}")