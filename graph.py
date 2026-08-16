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
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "web_search is unavailable: TAVILY_API_KEY is not set on the server."

    from tavily import TavilyClient

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=5)
        results = response.get("results", [])
    except Exception as exc:  # noqa: BLE001
        return f"Search failed: {exc}"

    if not results:
        return "No results found."

    lines = []
    for r in results:
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        lines.append(f"- {title}: {content} ({url})")
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


@tool
def fetch_url(url: str) -> str:
    """Fetch the raw content of a specific URL (e.g. an API's documentation
    page, or an API endpoint itself). Use this after web_search finds a
    promising page or API you need to read before calling it. Returns up
    to ~4000 characters of the response body."""
    import requests

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "agent/1.0"})
        resp.raise_for_status()
        return resp.text[:4000]
    except Exception as exc:  # noqa: BLE001
        return f"Could not fetch '{url}': {exc}"


@tool
def run_python(code: str) -> str:
    """Execute a snippet of Python code and return whatever it prints.
    Use this to call APIs you've discovered (e.g. with the `requests`
    library), parse data, or do anything a pre-built tool doesn't cover.
    Always `print()` the result you want to see — nothing is returned
    automatically. `requests`, `json`, `math`, and `datetime` are
    pre-imported and available.

    SECURITY NOTE: this runs on the same server as everything else, with
    no sandbox. Only use this on a deployment only you can access — never
    expose an agent with this tool to untrusted/public users."""
    import contextlib
    import io
    import json as _json
    import math as _math
    import requests as _requests

    safe_globals = {
        "__builtins__": __builtins__,
        "requests": _requests,
        "json": _json,
        "math": _math,
        "datetime": datetime,
    }

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, safe_globals, {})  # noqa: S102 — intentional, see docstring
        output = buffer.getvalue().strip()
        return output if output else "(code ran with no printed output)"
    except Exception as exc:  # noqa: BLE001
        return f"Error running code: {exc}"


TOOLS = [web_search, fetch_url, run_python, calculator, current_datetime]


# ---------------------------------------------------------------------------
# 3. System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful, thorough personal assistant agent
with the ability to discover and use new tools on the fly, not just the
ones pre-built for you.

- Use tools whenever they would make your answer more accurate, current,
  or verifiable — don't guess at facts you can look up or compute.
- If a task needs a capability you don't have a dedicated tool for
  (e.g. calling some specific API), don't say you can't do it — instead:
  1. Use web_search to find the right API or documentation.
  2. Use fetch_url to read its docs / endpoint format.
  3. Use run_python to write and execute the actual code that calls it,
     using the `requests` library, and print() the result.
- STRONGLY prefer APIs that need no signup and no API key (e.g. Open-Meteo
  for weather, Wikipedia's API, government open-data endpoints). Many free
  APIs are genuinely keyless — look for those first.
- If every option you find requires creating an account (email
  verification, a dashboard signup, a paid plan), STOP after one attempt.
  You cannot create accounts, verify emails, or solve CAPTCHAs — no
  amount of retrying will produce a working key. Tell the user plainly
  that the API requires a manual signup, give them the signup link, and
  say you'll use it as soon as they provide the key. Do not invent a
  placeholder key and run the code anyway — it will always fail and
  wastes the user's time.
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

