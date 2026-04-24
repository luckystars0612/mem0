# OpenMemory — MiniMax + Ollama Setup Guide

This document records all required changes to make OpenMemory work with **MiniMax** (LLM) and **Ollama** (Embedder) instead of the default OpenAI.

---

## 1. Install Ollama (native on Windows)

```powershell
winget install Ollama.Ollama
ollama pull nomic-embed-text
```

Ollama runs natively — no Docker needed. Verify:

```powershell
curl http://localhost:11434
# Expected: Ollama is running
```

---

## 2. Fix `docker-compose.yml`

**Bug:** Wrong volume path causes data loss on restart.

```yaml
# ❌ Wrong (default)
mem0_store:
  volumes:
    - mem0_storage:/mem0/storage

# ✅ Correct
mem0_store:
  volumes:
    - mem0_storage:/qdrant/storage
```

**UI environment** — hardcode instead of dynamic variables (avoids startup warnings):

```yaml
openmemory-ui:
  ports:
    - "3001:3000"   # change port as needed
  environment:
    - NEXT_PUBLIC_API_URL=http://localhost:8765
    - NEXT_PUBLIC_USER_ID=default_user
```

---

## 3. Fix `api/.env`

```env
OPENAI_API_KEY=fake-not-used
USER=default_user
API_KEY=fake-not-used

# LLM = MiniMax
LLM_PROVIDER=minimax
LLM_MODEL=MiniMax-M2.7
LLM_API_KEY=your_minimax_key_here
LLM_BASE_URL=https://api.minimax.io/v1

# Embedder = Ollama local
EMBEDDER_PROVIDER=ollama
EMBEDDER_MODEL=nomic-embed-text
EMBEDDER_BASE_URL=http://host.docker.internal:11434

# Vector store dimension (nomic-embed-text = 768)
EMBEDDING_DIMS=768
```

> **Note:** `.env` values are only applied when the DB is first created. They do not override an existing DB — see sections 4 and 7.

---

## 4. Fix `api/app/routers/config.py`

**Problem:** `get_default_configuration()` hardcodes OpenAI and does not read from `.env`. Config is stored in SQLite DB on first init and never re-read from env.

**Fix:** Add `import os` and `load_dotenv()` at the top of the file, then rewrite the function:

```python
import os
from dotenv import load_dotenv
load_dotenv()

def get_default_configuration():
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_base_url = os.getenv("LLM_BASE_URL", None)

    embedder_provider = os.getenv("EMBEDDER_PROVIDER", "openai")
    embedder_model = os.getenv("EMBEDDER_MODEL", "text-embedding-3-small")
    embedder_base_url = os.getenv("EMBEDDER_BASE_URL", None)

    llm_config = {
        "model": llm_model,
        "temperature": 0.1,
        "max_tokens": 2000,
        "api_key": "env:LLM_API_KEY",
    }
    if llm_base_url:
        llm_config["openai_base_url"] = llm_base_url

    return {
        "openmemory": {"custom_instructions": None},
        "mem0": {
            "llm": {
                "provider": llm_provider,
                "config": llm_config
            },
            "embedder": {
                "provider": embedder_provider,
                "config": {
                    "model": embedder_model,
                    "api_key": None,
                    "ollama_base_url": embedder_base_url
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "openmemory",
                    "host": "mem0_store",
                    "port": 6333,
                    "embedding_model_dims": int(os.getenv("EMBEDDING_DIMS", "768"))
                }
            }
        }
    }
```

> **Important:** `embedding_model_dims: 768` must be set to match `nomic-embed-text`. Without it, Qdrant creates the collection with dim 1536 (OpenAI default) causing a dimension mismatch error on every memory operation.

---

## 5. Fix `api/app/utils/categorization.py`

**Problem:** Hardcodes `OpenAI()` client and calls `api.openai.com` directly, ignoring the configured LLM. MiniMax also returns `<think>` reasoning tags when called via the OpenAI SDK.

**Fix:** Use `httpx` to call MiniMax's native `/v1/text/chatcompletion_v2` endpoint directly (no `<think>` tags):

```python
import json
import logging
import os
import re
from typing import List

import httpx
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
model = os.getenv("LLM_MODEL", "gpt-4o-mini")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
                {"role": "user", "content": memory}
            ],
            "temperature": 0,
        }

        # MiniMax: use /text/chatcompletion_v2 to avoid <think> tags
        # Other providers: use standard /chat/completions
        if "minimax.io" in base_url:
            endpoint = "https://api.minimax.io/v1/text/chatcompletion_v2"
        else:
            endpoint = f"{base_url}/chat/completions"

        with httpx.Client() as client:
            resp = client.post(endpoint, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]

        # Strip <think> tags just in case
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        # Strip markdown code blocks
        content = re.sub(r'```json|```', '', content).strip()

        parsed = json.loads(content)
        return [cat.strip().lower() for cat in parsed.get("categories", [])]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        raise
```

---

## 6. Fix `ui/entrypoint.sh` (CRLF line endings)

Windows git clone converts line endings to CRLF, which breaks the shell script inside the Linux container.

```powershell
$content = Get-Content ui/entrypoint.sh -Raw
$content = $content -replace "`r`n", "`n"
[System.IO.File]::WriteAllText("$PWD/ui/entrypoint.sh", $content)
```

---

## 7. Full Reset Procedure

When config changes are made or something breaks:

```powershell
docker compose down

# Delete SQLite DB (holds old config)
del api\openmemory.db

# Delete Qdrant collections (avoid dimension mismatch)
Invoke-RestMethod -Uri "http://localhost:6333/collections/openmemory" -Method DELETE
Invoke-RestMethod -Uri "http://localhost:6333/collections/mem0migrations" -Method DELETE

# Rebuild and start
docker compose build --no-cache openmemory-mcp
docker compose up
```

---

## 8. Connect Cline

Add to `cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "openmemory": {
      "url": "http://localhost:8765/mcp/cline/sse/default_user",
      "type": "sse"
    }
  }
}
```

Add to `.clinerules` in your project:

```
At the start of every task, search memory for relevant past context using the search_memory tool.
After completing any task, save key decisions, changes made, and important context using the add_memories tool.
```

---

## 9. Verify Everything Works

```powershell
# Create a test memory
Invoke-RestMethod -Uri "http://localhost:8765/api/v1/memories/" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"user_id":"default_user","text":"Test memory"}'

# Check current config
curl http://localhost:8765/api/v1/config/
```

Logs should show:
- `https://api.minimax.io/v1/text/chatcompletion_v2 HTTP/1.1 200 OK`
- `http://host.docker.internal:11434/api/embed HTTP/1.1 200 OK`
- No calls to `api.openai.com`

---

## Summary of Modified Files

| File | Problem | Fix |
|---|---|---|
| `docker-compose.yml` | Wrong Qdrant volume path, dynamic UI env vars | Fix path, hardcode env values |
| `api/.env` | No MiniMax/Ollama config | Add LLM/Embedder/Dims config |
| `api/app/routers/config.py` | Hardcoded OpenAI defaults, missing vector dims | Read from env, add `embedding_model_dims` |
| `api/app/utils/categorization.py` | Hardcoded OpenAI client, `<think>` tag pollution | Use httpx with MiniMax native endpoint |
| `ui/entrypoint.sh` | CRLF line endings on Windows | Convert to LF |