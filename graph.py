"""
Production-style LangGraph agent with a multi-provider fallback router.

Model order: Gemini 2.5 Flash (primary) -> Groq Llama 3.3 70B -> Groq GPT-OSS 120B
If the primary provider rate-limits (HTTP 429) or errors, LangChain's
.with_fallbacks() automatically retries on the next provider in the chain.
This is what lets a fully free stack absorb far more daily traffic than
any single provider's free tier would allow on its own.

This graph is designed to be served with `langgraph dev` / `langgraph up`
(the free, open-source LangGraph API server) so it plugs directly into
LangChain's official agent-chat-ui with zero LangSmith Platform usage
and therefore zero LCU consumption.
"""

import os
from datetime import datetime, timezone

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent


# ---------------------------------------------------------------------------
# 1. Model router: primary + fallbacks across free-tier providers
# ---------------------------------------------------------------------------

def build_llm():
    primary = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.4,
    )

    fallback_1 = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0.4,
    )

    fallback_2 = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0.4,
    )

    # exception_key=None means: fall through on ANY exception (429, timeout,
    # 5xx, etc.), not just rate limits. Order = priority.
    return primary.with_fallbacks([fallback_1, fallback_2])


# ---------------------------------------------------------------------------
# 2. Tools — swap these for whatever your agent actually needs
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the public web for current information and return top results
    as short text snippets. Use this whenever the user asks about something
    recent, time-sensitive, or outside general knowledge."""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return "No results found."

    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"- {title}: {body} ({href})")
    return "\n".join(lines)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(42 * 7) / 3'.
    Only supports numbers and + - * / ( ) . ** operators."""
    import ast
    import operator

    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as exc:  # noqa: BLE001
        return f"Could not evaluate '{expression}': {exc}"


@tool
def current_datetime() -> str:
    """Return the current UTC date and time. Use this instead of guessing
    the date."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


TOOLS = [web_search, calculator, current_datetime]


# ---------------------------------------------------------------------------
# 3. System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful, thorough personal assistant agent.

- Use tools whenever they would make your answer more accurate, current,
  or verifiable — don't guess at facts you can look up or compute.
- Think step by step for multi-part requests, and check your own work
  before answering.
- Be direct and concise. Skip unnecessary preamble.
- If a tool result is incomplete or ambiguous, say so rather than filling
  in gaps with assumptions.
"""

# ---------------------------------------------------------------------------
# 4. Build the graph
# ---------------------------------------------------------------------------

# No checkpointer is passed here on purpose: the LangGraph API server
# (langgraph dev / langgraph up) provides per-thread persistence
# automatically once deployed. Supplying your own checkpointer conflicts
# with that and is what caused the startup error you saw. If you need to
# point persistence at a specific Postgres database instead of the
# platform's default, set the POSTGRES_URI environment variable — no code
# change needed.
graph = create_react_agent(
    build_llm(),
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
)
