from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.callbacks import get_openai_callback
from urllib.parse import urlencode, urlparse
import sys
import asyncio
import logging
import json
from os import getenv
from dotenv import load_dotenv
from urllib.parse import urlencode

from utils.asserted_env import asserted_env
from claims import ClaimHandler
from utils.memory.local_memory import get_local_short_term_memory, upload_local_short_term_memory, InMemoryHistory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SLEEP_INTERVAL = 1
USD_PER_TOKEN = 0.000001

class HTTPFilter(logging.Filter):
    def filter(self, record):
        return not (record.msg and record.msg.startswith("HTTP Request:"))
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers:
    handler.addFilter(HTTPFilter())
logging.getLogger('requests').addFilter(HTTPFilter())

def get_tools_description(tools):
    return "\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )

async def create_agent(coral_tools, agent_tools):
    combined_tools = coral_tools + agent_tools

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are FirecrawlMCP-Agent, an AI agent designed to assist with web scraping, crawling, and content extraction tasks using Firecrawl.
            You have access to a variety of tools to help you scrape websites, crawl entire sites, search the web, extract structured data, and discover URLs on websites.

            CURRENT USER REQUEST:
            {user_request}

            CONVERSATION HISTORY:
            {history}

            AGENT INSTRUCTIONS:
            {coral_instruction}

            SYSTEM PROMPT:
            {system_prompt}
         
            EXTRA PROMPT:
            {extra_prompt}

            MESSAGES CONTEXT:
            {coral_messages}
                    """.strip()),
                    MessagesPlaceholder("agent_scratchpad"),
                ])

    model = init_chat_model(
        model=asserted_env("MODEL_NAME"),
        model_provider=asserted_env("MODEL_PROVIDER"),
        base_url=getenv("MODEL_BASE_URL"),
        api_key=asserted_env("MODEL_API_KEY"),
        temperature=float(asserted_env("MODEL_TEMPERATURE")),
        max_tokens=int(float(asserted_env("MODEL_MAX_TOKENS"))),
    )

    agent = create_tool_calling_agent(model, combined_tools, prompt)
    return AgentExecutor(agent=agent, tools=combined_tools, verbose=True, handle_parsing_errors=True)

async def main():
    global claim_handler
    claim_handler = ClaimHandler("micro_coral")

    CORAL_RUNTIME = getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if CORAL_RUNTIME is None:
        _ = load_dotenv()

    CORAL_SSE_URL = asserted_env("CORAL_SSE_URL")
    MAX_ITERATIONS = int(float(asserted_env("MAX_ITERATIONS")))
    TIMEOUT = float(getenv("TIMEOUT_MS", "300000"))
    MEMORY = getenv("MEMORY")

    coral_params = {"agentId": asserted_env("CORAL_AGENT_ID"), "agentDescription": getenv("AGENT_DESCRIPTION", None)}
    query_string = urlencode(coral_params)
    CORAL_SERVER_URL = f"{CORAL_SSE_URL}?{query_string}"

    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": CORAL_SERVER_URL,
                "timeout": TIMEOUT,
                "sse_read_timeout": TIMEOUT,
            },
            "firecrawl": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "firecrawl-mcp"],
                "env": {"FIRECRAWL_API_KEY": asserted_env("FIRECRAWL_API_KEY")},
            },
        }
    )

    logger.info(f"Connecting to Coral @ '{CORAL_SERVER_URL}'")
    async with client.session("coral") as coral_session:
        coral_tools = await client.get_tools(server_name="coral")
        logger.info(f"Coral tools count: {len(coral_tools)}")
        logger.info(f"Coral tools: {get_tools_description(coral_tools)}")
        coral_tools_list = {tool.name: tool for tool in coral_tools}

        agent_tools = await client.get_tools(server_name="firecrawl")
        logger.info(f"Agent tools count: {len(agent_tools)}")
        # logger.info(f"Agent tools: {get_tools_description(agent_tools)}")
        
        coral_instruction = (await load_mcp_resources(coral_session, uris="coral://agent/instruction"))[0]
        logger.info(f"Coral instruction: {coral_instruction.as_string()}")

        coral_session_id = f"{urlparse(CORAL_SERVER_URL).path.split('/')[-2]}"
        logger.info(f"Session ID: {coral_session_id}")

        agent_executor = await create_agent(coral_tools, agent_tools)

        for iteration in range(MAX_ITERATIONS):
            logger.info("*********************WAITING FOR INPUT, LOOP NUMBER: %d*********************", iteration + 1)
            if claim_handler.no_budget():
                logger.info("No more budget - breaking loop")
                break


            memory_history_str = await get_local_short_term_memory()
            logger.info(f"Loaded local memory history: {memory_history_str}")

            coral_messages = (await load_mcp_resources(coral_session, uris="coral://messages"))[0]
            logger.info(f"Coral messages: {coral_messages.as_string()}")

            user_request = await coral_tools_list['coral_wait_for_mentions'].ainvoke({
                "timeoutMs": 10000 
            })
            
            if isinstance(user_request, str) and 'error_timeout' in user_request:
                logger.info("Timeout - keep waiting...")
                await asyncio.sleep(SLEEP_INTERVAL)
                continue
            
            logger.info(f"User: {user_request}")

            with get_openai_callback() as cb:
                result = await agent_executor.ainvoke({
                    "user_request": user_request,
                    "history": memory_history_str,
                    "coral_instruction": coral_instruction.as_string(),
                    "coral_messages": coral_messages.as_string(),
                    "system_prompt": asserted_env("SYSTEM_PROMPT"),
                    "extra_prompt": getenv("CORAL_PROMPT_SYSTEM", ""),
                    "agent_scratchpad": [],
                })

            response = result["output"]
            logger.info(f"Agent response: {response}")

            total_to_claim = 0.0
            total_tokens = 0
            try:
                total_tokens = cb.total_tokens
                total_to_claim = total_tokens * USD_PER_TOKEN
                logger.info(f"Token usage: {total_tokens} → Claiming ${total_to_claim:.6f}")
            except AttributeError:
                total_to_claim = 0.0
                total_tokens = 0
                logger.warning("No response_metadata.token_usage.total_tokens"
                               " found on step result. Can't calculate token cost.")
            if total_to_claim > 0.0:
                claim_handler.claim(total_to_claim)
                logger.info(f"Claimed cost for step: ${total_to_claim:.6f} for {total_tokens} tokens")


            await upload_local_short_term_memory(
                    user_message=user_request,
                    assistant_response=response
                )

if __name__ == "__main__":
    asyncio.run(main())