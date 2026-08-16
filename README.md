# Free-stack LangChain agent — official chat UI + fallback model router

This gives you the real LangChain/LangGraph agent-building experience —
`create_react_agent`, tool calling, per-thread memory, streaming — served
through LangChain's actual official chat UI, running entirely on free-tier
infrastructure. No LangSmith Platform, no LCUs, no local GPU.

**Two pieces:**
1. `backend/` — a LangGraph agent (this repo) served by `langgraph dev`,
   the free open-source dev server.
2. `agent-chat-ui` — LangChain's official Next.js chat frontend (you clone
   it separately, see below), pointed at your backend.

Your only real ceiling is the LLM providers' free rate limits, since the
model call falls through Gemini → Groq → Groq automatically on a 429.

---

## 1. Get free API keys (2 minutes, no credit card)

- **Gemini**: https://aistudio.google.com/apikey
- **Groq**: https://console.groq.com/keys

Copy `backend/.env.example` to `backend/.env` and paste both keys in.

## 2. Run the backend locally first (sanity check)

```bash
cd backend
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
langgraph dev
```

This starts the LangGraph API server on `http://localhost:2024` and opens
LangGraph Studio in your browser — a visual debugger for the graph, free,
part of the open-source CLI (also not a LangSmith Platform product).

## 3. Get the official chat UI running

LangChain's actual chat frontend is a separate open-source repo:

```bash
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
npm install
npm run dev
```

On first load it asks for:
- **Deployment URL**: `http://localhost:2024` (your local backend)
- **Assistant/Graph ID**: `agent` (matches `langgraph.json`)
- **LangSmith API key**: leave blank — not required for a self-hosted graph

You now have the identical chat UI/UX to LangChain's own product, talking
to your agent, 100% free, 100% local.

---

## 4. Deploy for real (still free)

### Backend → Render
1. Push the `backend/` folder to a GitHub repo.
2. On Render: **New → Web Service**, connect the repo, pick the free
   instance type. Render detects the `Dockerfile` automatically.
3. Add `GOOGLE_API_KEY` and `GROQ_API_KEY` as environment variables in
   Render's dashboard (don't commit `.env`).
4. Deploy. You'll get a URL like `https://your-agent.onrender.com`.

Free-tier note: Render's free web services sleep after inactivity and take
~30–60s to wake on the next request. Fine for personal use; not for a
product with real-time SLA expectations.

### Frontend → Vercel
1. Push your `agent-chat-ui` clone to its own GitHub repo (or fork it
   directly on GitHub, then import that fork).
2. On Vercel: **Add New → Project**, import the repo, deploy — it
   auto-detects Next.js.
3. In Vercel's project settings, set the env var
   `NEXT_PUBLIC_API_URL` to your Render backend URL.
4. Redeploy. Your chat UI is now live at `https://your-project.vercel.app`.

Both tiers are free indefinitely for this scale of personal use — no
trial period, no card required on either platform for the free tier.

---

## 5. Why this doesn't need LCUs at all

LCUs are a LangSmith/LangGraph **Platform** billing meter — they only
apply if LangChain Inc. hosts your deployment for you. Here, Render is
hosting the LangGraph server and Vercel is hosting the UI — LangChain's
infra isn't involved, so there's nothing to meter. `langgraph dev`/`up`
and `agent-chat-ui` are MIT-licensed open source either way.

If you later want tracing/eval dashboards, the free LangSmith Developer
tier (5,000 traces/month) works alongside this setup — just uncomment the
`LANGCHAIN_*` vars in `.env`. It's optional and doesn't gate usage.

---

## 6. Extending the agent

- Add tools: define a new `@tool`-decorated function in `graph.py`,
  append it to the `TOOLS` list.
- Add/reorder fallback models: edit `build_llm()` — any LangChain chat
  model class works in the fallback chain (OpenRouter, Cerebras,
  HuggingFace, etc.), so you can keep stacking free tiers as you find
  more.
- Persistent memory across restarts: swap `MemorySaver()` for
  `PostgresSaver` (needs a free Postgres instance, e.g. Supabase/Neon
  free tier) — same interface, just durable.
