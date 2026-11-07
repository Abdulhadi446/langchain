"""Answer a question, fact-check the answer, and retry when needed.

Run with:
    python main.py

Required environment variables:
    None. Sodeom documents that the OpenAI-compatible client accepts any key.
    OPENAI_MODEL (optional, defaults to beam)
    OPENAI_BASE_URL (optional, defaults to https://sodeom.com/v1)
"""

from __future__ import annotations
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class FactCheck(BaseModel):
    """The structured result produced by the fact-checking agent."""

    verdict: Literal["true", "false"] = Field(
        description="Whether the answer is sufficiently supported and accurate."
    )
    feedback: str = Field(
        description="Specific corrections the answer agent should make if false."
    )


class WorkflowState(TypedDict, total=False):
    """The state of the workflow."""

    question: str
    answer: str
    feedback: str
    verdict: Literal["true", "false"]
    attempts: int


load_dotenv()


def build_workflow(max_attempts: int = 3):
    """Build the answer/fact-check workflow with a bounded retry loop."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "beam"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://sodeom.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "any"),
        temperature=0,
    )
    fact_checker = model.with_structured_output(
        FactCheck,
        method="function_calling",
    )

    def answer_agent(state: WorkflowState) -> dict[str, object]:
        retry_context = ""
        if state.get("feedback"):
            retry_context = (
                "\nA previous draft was fact-checked as incorrect. "
                f"Fix these issues:\n{state['feedback']}\n"
            )

        response = model.invoke(
            [
                (
                    "system",
                    "You are a careful answer agent. Answer the user's question "
                    "accurately and clearly. Do not invent facts. If the question "
                    "depends on current information, state any uncertainty.",
                ),
                (
                    "human",
                    f"Question: {state['question']}\n"
                    f"{retry_context}\n"
                    "Return only the answer text.",
                ),
            ]
        )
        return {
            "answer": response.content,
            "attempts": state.get("attempts", 0) + 1,
        }

    def fact_check_agent(state: WorkflowState) -> dict[str, object]:
        result = fact_checker.invoke(
            [
                (
                    "system",
                    "You are a strict fact-checking agent. Check the draft against "
                    "your knowledge and the question. Mark it true only when it is "
                    "accurate enough to send to the user. Explain every needed "
                    "correction in feedback.",
                ),
                (
                    "human",
                    f"Question: {state['question']}\n"
                    f"Draft answer:\n{state['answer']}",
                ),
            ]
        )
        return {"verdict": result.verdict, "feedback": result.feedback}

    def route_after_fact_check(state: WorkflowState) -> str:
        if state.get("verdict") == "true":
            return "finish"
        if state.get("attempts", 0) >= max_attempts:
            return "finish"
        return "retry"

    graph = StateGraph(WorkflowState)
    graph.add_node("answer", answer_agent)
    graph.add_node("fact_check", fact_check_agent)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", "fact_check")
    graph.add_conditional_edges(
        "fact_check",
        route_after_fact_check,
        {"retry": "answer", "finish": END},
    )
    return graph.compile()


def answer_question(question: str, max_attempts: int = 3) -> WorkflowState:
    """Run the workflow and return the final state."""
    if not question.strip():
        raise ValueError("question must not be empty")
    return build_workflow(max_attempts).invoke(
        {"question": question.strip(), "attempts": 0}
    )


if __name__ == "__main__":
    question = input("Ask a question: ").strip()
    result = answer_question(question)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nFact-check verdict: {result['verdict']}")
    print(f"Attempts: {result['attempts']}")
    if result["verdict"] == "false":
        print(
            "\nWarning: the retry limit was reached before the fact checker "
            "approved the answer."
        )
