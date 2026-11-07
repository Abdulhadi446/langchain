# Fact-checked LangChain answer agent

This template runs two agents:

1. An answer agent drafts an answer.
2. A fact-checking agent returns a structured `true`/`false` verdict.
3. If the verdict is `false`, the answer agent retries with the fact-checker's
   corrections.

The workflow stops when the answer is approved or after three attempts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Sodeom accepts any placeholder value as OPENAI_API_KEY
python main.py
```

The client uses the OpenAI-compatible endpoint `https://sodeom.com/v1` by
default and uses the `beam` model. Set `OPENAI_MODEL` to another supported model
if needed.
You can override the endpoint with `OPENAI_BASE_URL`.

## Workflow diagram

The draw.io diagram shows the answer, fact-check, retry, approval, and
maximum-attempts paths:

[Open workflow.drawio](./workflow.drawio)
