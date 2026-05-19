import sys
import types

import pytest


def _install_optional_evaluation_dependency_stubs() -> None:
    # These focused unit tests do not load datasets or run external metrics.
    # Stubbing keeps this boundary test independent of the heavyweight eval stack.
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None
    datasets.get_dataset_config_names = lambda *args, **kwargs: []
    sys.modules["datasets"] = datasets

    math_verify = types.ModuleType("math_verify")
    math_verify.parse = lambda value: value
    math_verify.verify = lambda expected, actual: expected == actual
    sys.modules["math_verify"] = math_verify

    lm_eval = types.ModuleType("lm_eval")
    lm_eval.__path__ = []
    tasks = types.ModuleType("lm_eval.tasks")
    tasks.__path__ = []
    drop = types.ModuleType("lm_eval.tasks.drop")
    drop.__path__ = []
    drop_utils = types.ModuleType("lm_eval.tasks.drop.utils")
    drop_utils.process_results = lambda *args, **kwargs: {}
    drop_utils.process_docs = lambda docs: docs

    sys.modules["lm_eval"] = lm_eval
    sys.modules["lm_eval.tasks"] = tasks
    sys.modules["lm_eval.tasks.drop"] = drop
    sys.modules["lm_eval.tasks.drop.utils"] = drop_utils


_install_optional_evaluation_dependency_stubs()

from evaluation.benchmarks import (  # noqa: E402
    StandardMCQBenchmark,
    _arc_row_to_doc,
    _choice_label_to_index,
    _format_mcq,
)


def _numeric_arc_row() -> dict:
    return {
        "question": "Rocks are classified according to what?",
        "choices": {
            "text": [
                "their color",
                "their shape",
                "how they formed",
                "the minerals they contain",
            ],
            "label": ["1", "2", "3", "4"],
        },
        "answerKey": "3",
    }


def _letter_arc_row() -> dict:
    return {
        "question": "Which object best conducts electricity?",
        "choices": {
            "text": ["rubber band", "copper wire", "glass cup", "wood block"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
    }


def test_arc_numeric_choice_labels_become_canonical_gold_index() -> None:
    doc = _arc_row_to_doc(_numeric_arc_row())

    assert doc.gold_index == 2
    assert doc.choices == _numeric_arc_row()["choices"]["text"]


def test_arc_numeric_choice_gold_formats_and_scores_as_canonical_letter() -> None:
    doc = _arc_row_to_doc(_numeric_arc_row())
    formatted, choices = _format_mcq(doc, include_gold=True)

    assert choices == ["A", "B", "C", "D"]
    assert "A. their color" in formatted
    assert "C. how they formed" in formatted
    assert formatted.endswith("Answer: C")

    benchmark = StandardMCQBenchmark()
    benchmark.ground_truths = [{"valid_set": set(choices), "gold": "C"}]

    assert benchmark.compute_metrics(["C"]) == {"n": 1, "acc": 1.0, "invalid": 0.0}


def test_arc_letter_choice_labels_still_map_through_row_converter() -> None:
    assert _arc_row_to_doc(_letter_arc_row()).gold_index == 1


def test_choice_label_to_index_reports_missing_answer_key() -> None:
    with pytest.raises(ValueError) as exc_info:
        _choice_label_to_index(["A", "B", "C", "D"], "E")

    message = str(exc_info.value)
    assert "answerKey 'E'" in message
    assert "['A', 'B', 'C', 'D']" in message
