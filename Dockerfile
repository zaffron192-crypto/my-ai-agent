FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT at runtime; langgraph dev serves the same API shape
# that agent-chat-ui and the LangGraph SDK expect, with no LangSmith
# Platform involvement (no LCUs).
CMD ["sh", "-c", "langgraph dev --host 0.0.0.0 --port ${PORT:-8000} --no-browser"]
