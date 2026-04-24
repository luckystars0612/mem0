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
base_url = os.getenv("LLM_BASE_URL", "https://api.minimax.io/v1").rstrip("/")
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

        # Dùng /text/chatcompletion_v2 cho MiniMax để tránh <think> tags
        # Fallback sang /chat/completions cho các provider khác
        if "minimax.io" in base_url:
            endpoint = "https://api.minimax.io/v1/text/chatcompletion_v2"
        else:
            endpoint = f"{base_url}/chat/completions"

        with httpx.Client() as client:
            resp = client.post(endpoint, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]

        # Strip <think> tags phòng trường hợp vẫn có
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        # Strip markdown code blocks
        content = re.sub(r'```json|```', '', content).strip()

        parsed = json.loads(content)
        return [cat.strip().lower() for cat in parsed.get("categories", [])]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        raise