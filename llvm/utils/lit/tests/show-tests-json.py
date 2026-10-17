# RUN: %{python} %s
# END.

import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from lit.BooleanExpression import BooleanExpression
from lit.formats.shtest import ShTest
from lit.Test import Test, TestSuite
from lit.main import build_test_inventory


INPUTS = os.path.join(os.path.dirname(__file__), "Inputs")
SUITE = os.path.join(INPUTS, "show-tests-json")


def run_lit(*args):
    env = os.environ.copy()
    for variable in ("LIT_OPTS", "LIT_XFAIL", "LIT_XFAIL_NOT", "LIT_UNSUPPORTED"):
        env.pop(variable, None)
    return subprocess.run(
        [sys.executable, "-c", "from lit.main import main; main()", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def evaluate_tree(tree, features):
    if isinstance(tree, bool):
        return tree
    if isinstance(tree, str):
        return BooleanExpression.evaluate(tree, features)
    op = tree["op"]
    if op == "not":
        return not evaluate_tree(tree["operand"], features)
    if op == "and":
        return all(evaluate_tree(child, features) for child in tree["operands"])
    if op == "or":
        return any(evaluate_tree(child, features) for child in tree["operands"])
    raise AssertionError("unexpected expression operator: " + op)


class ShowTestsJsonTest(unittest.TestCase):
    def inventory(self, *args):
        result = run_lit("--show-tests-json", SUITE, *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_discovery_and_metadata(self):
        result = self.inventory("--filter=does-not-match", "--max-tests=1")
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(set(result), {"schema_version", "suites"})
        suites = result["suites"]
        self.assertEqual(len(suites), 3)
        self.assertEqual(
            [suite["name"] for suite in suites],
            ["json-inventory", "same-name", "same-name"],
        )
        self.assertEqual(
            [suite["source_root"] for suite in suites],
            [
                os.path.abspath(SUITE),
                os.path.abspath(os.path.join(SUITE, "same-a")),
                os.path.abspath(os.path.join(SUITE, "same-b")),
            ],
        )
        self.assertEqual(
            [suite["exec_root"] for suite in suites],
            [suite["source_root"] for suite in suites],
        )
        self.assertTrue(
            all(
                set(suite) == {"name", "source_root", "exec_root", "tests"}
                for suite in suites
            )
        )
        tests = suites[0]["tests"]
        self.assertEqual(len(tests), 3)
        self.assertEqual(
            [test["path_in_suite"] for test in tests],
            [
                "empty.test",
                "known.test",
                os.path.join("unknown", "binary.test"),
            ],
        )
        known = tests[1]
        self.assertEqual(set(known), {"path_in_suite", "requires"})
        self.assertEqual(
            known["requires"],
            {
                "op": "and",
                "operands": [
                    "a",
                    "preexisting",
                    "z",
                    {"op": "not", "operand": "disabled"},
                    {
                        "op": "or",
                        "operands": ["b", "feature={{[a-z]+}}"],
                    },
                ],
            },
        )
        self.assertEqual(tests[0]["requires"], {})
        self.assertIsNone(tests[2]["requires"])
        self.assertEqual(
            [suite["tests"] for suite in suites[1:]],
            [
                [{"path_in_suite": "test.test", "requires": "a"}],
                [{"path_in_suite": "test.test", "requires": "b"}],
            ],
        )

    def test_exported_conditions_preserve_feature_semantics(self):
        requires = self.inventory("--filter=does-not-match")["suites"][0]["tests"][1][
            "requires"
        ]
        expressions = [
            "preexisting",
            "z && (a && z)",
            "feature={{[a-z]+}} || b",
            "!disabled && preexisting",
        ]
        for features in (
            (),
            ("a", "z", "preexisting", "feature=foo"),
            ("a", "z", "preexisting", "b", "disabled"),
            ("a", "z", "preexisting", "feature=123"),
            ("a", "z", "preexisting", "b"),
        ):
            with self.subTest(features=features):
                self.assertEqual(
                    evaluate_tree(requires, features),
                    all(
                        BooleanExpression.evaluate(expr, features)
                        for expr in expressions
                    ),
                )
                self.assertEqual(
                    evaluate_tree(requires, features),
                    set(("a", "z", "preexisting")).issubset(features)
                    and "disabled" not in features
                    and ("b" in features or "feature=foo" in features),
                )

    def test_separate_exec_root_and_source_paths(self):
        suite = os.path.join(INPUTS, "exec-discovery")
        result = run_lit("--show-tests-json", suite)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = next(
            suite
            for suite in json.loads(result.stdout)["suites"]
            if suite["name"] == "top-level-suite"
        )
        self.assertIn(
            os.path.join("subdir", "test-three.py"),
            [test["path_in_suite"] for test in record["tests"]],
        )
        test = next(
            test for test in record["tests"] if test["path_in_suite"] == "test-one.txt"
        )
        self.assertEqual(test["requires"], {})
        self.assertEqual(
            record["source_root"],
            os.path.abspath(os.path.join(INPUTS, "discovery")),
        )
        self.assertEqual(record["exec_root"], os.path.abspath(suite))

    def test_deterministic_output_and_response_file(self):
        first = run_lit("@" + os.path.join(SUITE, "options.rsp"), SUITE)
        second = run_lit("@" + os.path.join(SUITE, "options.rsp"), SUITE)
        reordered = run_lit(
            "--show-tests-json",
            *(
                os.path.join(SUITE, subdir, filename)
                for subdir, filename in (
                    ("same-b", "test.test"),
                    ("unknown", "binary.test"),
                    ("", "known.test"),
                    ("", "empty.test"),
                    ("same-a", "test.test"),
                )
            ),
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(reordered.returncode, 0, reordered.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stdout, reordered.stdout)
        self.assertEqual(
            first.stdout,
            json.dumps(json.loads(first.stdout), indent=2, sort_keys=True) + "\n",
        )

    def test_deprecated_order_option_preserves_warning(self):
        result = run_lit("--show-tests-json", "--incremental", SUITE)
        self.assertEqual(result.returncode, 0, result.stderr)
        warning, _, inventory = result.stdout.partition("\n")
        self.assertEqual(
            warning,
            "WARNING: --incremental is deprecated. Failing tests now always run first.",
        )
        self.assertEqual(json.loads(inventory)["schema_version"], 1)

    def test_invalid_metadata_has_diagnostic_without_partial_json(self):
        result = run_lit(
            "--show-tests-json",
            os.path.join(INPUTS, "show-tests-json-invalid", "bad.test"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("invalid-annotations :: bad.test", result.stderr)
        self.assertIn("a &&", result.stderr)

    def test_invalid_regex_has_diagnostic(self):
        result = run_lit(
            "--show-tests-json",
            os.path.join(INPUTS, "show-tests-json-invalid", "bad-regex.test"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("invalid-annotations :: bad-regex.test", result.stderr)
        self.assertIn("unterminated character set", result.stderr)

    def test_empty_boolean_continuation_has_test_specific_diagnostic(self):
        result = run_lit(
            "--show-tests-json",
            os.path.join(INPUTS, "show-tests-json-invalid", "bad-continuation.test"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("invalid-annotations :: bad-continuation.test", result.stderr)
        self.assertIn("Empty continuation in boolean expression", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_deep_boolean_expression_has_test_specific_diagnostic(self):
        config = SimpleNamespace(name="json-inventory", test_format=ShTest())
        suite = TestSuite(config.name, SUITE, SUITE, config)
        test = Test(suite, ["known.test"], config)
        test.requires.append("!" * 2000 + "a")
        with self.assertRaisesRegex(
            ValueError, "json-inventory :: known.test"
        ) as error:
            build_test_inventory([test])
        self.assertIn(
            os.path.abspath(os.path.join(SUITE, "known.test")), str(error.exception)
        )
        self.assertIsInstance(error.exception.__cause__, RecursionError)

    def test_duck_typed_format_without_metadata_is_unknown(self):
        config = SimpleNamespace(name="duck", test_format=object())
        suite = TestSuite("other-name", SUITE, SUITE, config)
        test = Test(suite, ["virtual", "case"], config, file_path="elsewhere")
        record = build_test_inventory([test])["suites"][0]
        self.assertEqual(
            record,
            {
                "name": "duck",
                "source_root": SUITE,
                "exec_root": SUITE,
                "tests": [
                    {
                        "path_in_suite": os.path.join("virtual", "case"),
                        "requires": None,
                    }
                ],
            },
        )

    def test_empty_logical_path_and_known_empty_requirements(self):
        config = SimpleNamespace(name="virtual", test_format=ShTest())
        suite = TestSuite(config.name, SUITE, SUITE, config)
        # Use a duck format for the empty virtual path: ShTest needs a source file.
        empty = Test(suite, (), SimpleNamespace(test_format=object()))
        ordinary = Test(suite, ("empty.test",), config)
        tests = build_test_inventory([ordinary, empty])["suites"][0]["tests"]
        self.assertEqual(
            tests,
            [
                {"path_in_suite": "", "requires": None},
                {"path_in_suite": "empty.test", "requires": {}},
            ],
        )

    def test_literal_true_and_boolean_identities(self):
        config = SimpleNamespace(name="bool", test_format=ShTest())
        suite = TestSuite(config.name, SUITE, SUITE, config)
        test = Test(suite, ("empty.test",), config)
        test.requires.append("true")
        requires = build_test_inventory([test])["suites"][0]["tests"][0]["requires"]
        self.assertEqual(requires, "true")
        self.assertNotIsInstance(requires, bool)
        self.assertTrue(evaluate_tree(requires, ()))
        self.assertEqual(BooleanExpression.combine("and", []), True)
        self.assertEqual(BooleanExpression.combine("or", []), False)
        self.assertEqual(BooleanExpression.combine("and", ["a", True, "a"]), "a")
        self.assertEqual(BooleanExpression.combine("or", ["a", False, "a"]), "a")
        self.assertEqual(BooleanExpression.combine("and", ["a", False]), False)
        self.assertEqual(BooleanExpression.combine("or", ["a", True]), True)
        self.assertEqual(
            BooleanExpression.combine(
                "and", ["true", True, {"op": "not", "operand": "a"}]
            ),
            {"op": "and", "operands": ["true", {"op": "not", "operand": "a"}]},
        )

    def test_unsatisfiable_requirements_are_not_empty(self):
        config = SimpleNamespace(name="contradiction", test_format=ShTest())
        suite = TestSuite(config.name, SUITE, SUITE, config)
        test = Test(suite, ("empty.test",), config)
        test.requires.append("a && !a")
        requires = build_test_inventory([test])["suites"][0]["tests"][0]["requires"]
        self.assertEqual(
            requires,
            {"op": "and", "operands": ["a", {"op": "not", "operand": "a"}]},
        )
        self.assertFalse(evaluate_tree(requires, ()))
        self.assertFalse(evaluate_tree(requires, ("a",)))

    def test_json_takes_precedence_over_inspection_and_result_options(self):
        expected = self.inventory()
        for option in (
            "--show-tests",
            "--show-suites",
            "--show-used-features",
            "-o",
            "--xunit-xml-output",
            "--resultdb-output",
            "--time-trace-output",
            "--wtt-output",
        ):
            with self.subTest(
                option=option
            ), tempfile.TemporaryDirectory() as directory:
                report_path = os.path.join(directory, "unused-report.json")
                args = [option]
                if option in (
                    "-o",
                    "--xunit-xml-output",
                    "--resultdb-output",
                    "--time-trace-output",
                    "--wtt-output",
                ):
                    args.append(report_path)
                result = run_lit("--show-tests-json", *args, SUITE)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), expected)
                self.assertFalse(os.path.exists(report_path))

    def test_text_inspection_unchanged(self):
        result = run_lit("--show-tests", SUITE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-- Available Tests --", result.stdout)
        self.assertNotIn('"schema_version"', result.stdout)

    def test_parsing_does_not_mutate_prepopulated_annotations(self):
        test = SimpleNamespace(
            getSourcePath=lambda: os.path.join(SUITE, "known.test"),
            requires=["preexisting"],
            unsupported=["disabled"],
            xfails=["old-xfail"],
        )
        requirements = ShTest().getTestRequirements(test)
        self.assertEqual(
            requirements,
            [
                "preexisting",
                "z && (a && z)",
                "feature={{[a-z]+}} || b",
                "!disabled && preexisting",
            ],
        )
        self.assertEqual(test.requires, ["preexisting"])
        self.assertEqual(test.unsupported, ["disabled"])
        self.assertEqual(test.xfails, ["old-xfail"])


class ExpressionTreeTest(unittest.TestCase):
    def test_structural_normalization(self):
        left = BooleanExpression.normalize("(b && a) && (a && b)")
        self.assertEqual(
            left,
            {
                "op": "and",
                "operands": ["a", "b"],
            },
        )
        self.assertEqual(
            BooleanExpression.normalize("b || (a || b)"),
            {
                "op": "or",
                "operands": ["a", "b"],
            },
        )
        self.assertEqual(
            BooleanExpression.normalize("!a && (b || feature={{[a-z]+}})"),
            {
                "op": "and",
                "operands": [
                    {"op": "not", "operand": "a"},
                    {
                        "op": "or",
                        "operands": ["b", "feature={{[a-z]+}}"],
                    },
                ],
            },
        )
        self.assertEqual(BooleanExpression.normalize("true"), "true")
        self.assertTrue(evaluate_tree(BooleanExpression.normalize("true"), ()))
        self.assertFalse(evaluate_tree(BooleanExpression.normalize("true-ish"), ()))
        self.assertNotEqual(
            BooleanExpression.normalize("a && (b || c)"),
            BooleanExpression.normalize("(a && b) || (a && c)"),
        )

    def test_combination_preserves_order_independent_normal_form(self):
        expressions = ["b", "a && b", "!c", "a"]
        forward = BooleanExpression.combine(
            "and", [BooleanExpression.normalize(expr) for expr in expressions]
        )
        backward = BooleanExpression.combine(
            "and", [BooleanExpression.normalize(expr) for expr in reversed(expressions)]
        )
        self.assertEqual(forward, backward)
        self.assertEqual(
            forward,
            {
                "op": "and",
                "operands": ["a", "b", {"op": "not", "operand": "c"}],
            },
        )

    def test_syntax_error(self):
        with self.assertRaisesRegex(ValueError, "in expression: 'a &&'"):
            BooleanExpression.normalize("a &&")


if __name__ == "__main__":
    unittest.main()
