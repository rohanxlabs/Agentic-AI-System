"""Groq LLM integration module."""
import os
import logging
import time
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

from config.config import MODEL_NAME

load_dotenv()
logger = logging.getLogger(__name__)


class GroqLLM:
    """Interface to Groq language model API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize Groq LLM client.

        Args:
            api_key: Groq API key (uses env var if not provided)

        Raises:
            ValueError: If API key is not available
        """
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=key)

    def call(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """Call the Groq API with a prompt.

        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length

        Returns:
            Model response text

        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            # Add delay to prevent rate limiting
            time.sleep(2.0)  # 2 second delay between API calls
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise