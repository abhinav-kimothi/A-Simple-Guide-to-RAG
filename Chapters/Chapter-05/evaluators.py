from typing import Optional

from langchain.evaluation import load_evaluator
from langchain.smith import RunEvalConfig
from langchain_openai import ChatOpenAI

try:
    from langchain.schema.language_model import BaseLanguageModel
except ImportError:
    from langchain_core.language_models import BaseLanguageModel
from langsmith.evaluation.evaluator import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run


# TODO: Split this into an assertion-by-assertion evaluator
# TODO: Combine with a document relevance evaluator (to report retriever performance)
class FaithfulnessEvaluator(RunEvaluator):
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        self.evaluator = load_evaluator(
            "labeled_score_string",
            criteria={
                "faithfulness": """
Score 1: The answer directly contradicts the information provided in the reference docs.
Score 3: The answer contains a mix of correct information from the reference docs and incorrect or unverifiable information not found in the docs.
Score 5: The answer is mostly aligned with the reference docs but includes extra information that, while not contradictory, is not verified by the docs.
Score 7: The answer aligns well with the reference docs but includes minor, commonly accepted facts not found in the docs.
Score 10: The answer perfectly aligns with and is fully entailed by the reference docs, with no extra information."""
            },
            llm=llm,
            normalize_by=10,
        )

    @staticmethod
    def _get_retrieved_docs(run: Run) -> str:
        # This assumes there is only one retriever in your chain.
        # To select more precisely, name your retrieval chain
        # using with_config(name="my_unique_name") and look up
        # by run.name
        runs = [run]
        while runs:
            run = runs.pop()
            if run.run_type == "retriever":
                return str(run.outputs["documents"])
            if run.child_runs:
                runs.extend(run.child_runs[::-1])
        return ""

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        try:
            docs_string = self._get_retrieved_docs(run)
            docs_string = f"Reference docs:\n<DOCS>\n{docs_string}\n</DOCS>\n\n"
            print(f"\n{docs_string[10]}\n")
            input_query = run.inputs["Question"]
            print(f"\nInput Query={input_query}\n")
            if run.outputs is not None and len(run.outputs) == 1:
                prediction = next(iter(run.outputs.values()))
                print(f"\nPrediction={prediction}\n")
            else:
                prediction = run.outputs["output"]
                print(f"\nPrediction={prediction}\n")
            result = self.evaluator.evaluate_strings(
                input=input_query,
                prediction=prediction,
                reference=docs_string,
            )
            return EvaluationResult(
                **{"key": "faithfulness", "comment": result.get("reasoning"), **result}
            )
        except Exception as e:
            return EvaluationResult(key="faithfulness", score=None, comment=repr(e))


_ACCURACY_CRITERION = {
    "accuracy": """
Score 1: The answer is incorrect and unrelated to the question or reference document.
Score 3: The answer shows slight relevance to the question or reference document but is largely incorrect.
Score 5: The answer is partially correct but has significant errors or omissions.
Score 7: The answer is mostly correct with minor errors or omissions, and aligns with the reference document.
Score 10: The answer is correct, complete, and perfectly aligns with the reference document.

If the reference answer contains multiple alternatives, the predicted answer must only match one of the alternatives to be considered correct.
If the predicted answer contains additional helpful and accurate information that is not present in the reference answer, it should still be considered correct.
"""  # noqa
}


def get_eval_config() -> RunEvalConfig:
    """Returns the evaluator for the environment."""
    eval_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        seed=42,
        max_retries=1,
        request_timeout=60,
    )
    # Use a longer-context LLM to check documents
    faithfulness_eval_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        seed=42,
        max_retries=1,
        request_timeout=60,
    )

    return RunEvalConfig(
        evaluators=[
            RunEvalConfig.LabeledScoreString(
                criteria=_ACCURACY_CRITERION, llm=eval_llm, normalize_by=10.0
            ),
            RunEvalConfig.EmbeddingDistance(),
        ],
        custom_evaluators=[FaithfulnessEvaluator(llm=faithfulness_eval_llm)],
    )
