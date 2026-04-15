"""Critic agent for analyzing and improving outputs."""
from typing import Any
from agents.base_agent import BaseAgent


class CriticAgent(BaseAgent):
    """Agent responsible for critical analysis and quality improvement."""

    def critique(self, output: str) -> str:
        """Provide critical analysis of output.
        
        Args:
            output: The output to critique
            
        Returns:
            Critique with identified flaws and improvements
        """
        prompt = f"""
Critically analyze this output.
List flaws, missing parts, and improvements.

Output:
{output}
"""
        return self.think(prompt)