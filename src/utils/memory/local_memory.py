# utils/memory/local_memory.py

from typing import Sequence
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

class InMemoryHistory:
    """
    In-memory implementation of chat message history. 
    Stores LangChain BaseMessage objects.
    """
    messages: list[BaseMessage]

    def __init__(self):
        self.messages = []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add a list of messages to the store."""
        self.messages.extend(messages)

    def add_message(self, message: BaseMessage):
        """Add a single message to the store."""
        self.messages.append(message)

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []

# Global instance of the memory for the local implementation
_local_history = InMemoryHistory()

async def get_local_short_term_memory() -> str:
    """Retrieves all messages from the in-memory history and formats them into a string."""
    history_lines = []
    for message in _local_history.messages:
        if isinstance(message, HumanMessage):
            history_lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            if message.content.strip():
                history_lines.append(f"Assistant: {message.content}")
    
    return "\n".join(history_lines)

async def upload_local_short_term_memory(user_message: str, assistant_response: str) -> None:
    """
    Adds the latest user and assistant messages to the in-memory history as BaseMessage objects.
    """
    _local_history.add_message(HumanMessage(content=user_message))
    _local_history.add_message(AIMessage(content=assistant_response))