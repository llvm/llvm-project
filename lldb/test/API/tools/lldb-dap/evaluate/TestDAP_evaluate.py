"""
Test lldb-dap completions request
"""

import re

import lldbdap_testcase
from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import line_number


class TestDAP_evaluate(lldbdap_testcase.DAPTestCaseBase):
    def assertEvaluate(self, expression, regex):
        self.assertRegex(
            self.dap_server.request_evaluate(expression, context=self.context)["body"][
                "result"
            ],
            regex,
        )

    def assertEvaluateFailure(self, expression):
        self.assertNotIn(
            "result",
            self.dap_server.request_evaluate(expression, context=self.context)["body"],
        )

    def isResultExpandedDescription(self):
        return self.context == "repl"

    def isExpressionParsedExpected(self):
        return self.context != "hover"

    def run_test_evaluate_expressions(
        self, context=None, enableAutoVariableSummaries=False
    ):
        """
        Tests the evaluate expression request at different breakpoints
        """
        self.context = context
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(
            program,
            enableAutoVariableSummaries=enableAutoVariableSummaries,
        )
        source = "main.cpp"
        breakpoint_lines = [
            line_number(source, "// breakpoint 1"),
            line_number(source, "// breakpoint 2"),
            line_number(source, "// breakpoint 3"),
            line_number(source, "// breakpoint 4"),
            line_number(source, "// breakpoint 5"),
            line_number(source, "// breakpoint 6"),
            line_number(source, "// breakpoint 7"),
            line_number(source, "// breakpoint 8"),
        ]
        breakpoint_ids = self.set_source_breakpoints(source, breakpoint_lines)

        self.assertEqual(
            len(breakpoint_ids),
            len(breakpoint_lines),
            "Did not resolve all the breakpoints.",
        )
        breakpoint_1 = breakpoint_ids[0]
        breakpoint_2 = breakpoint_ids[1]
        breakpoint_3 = breakpoint_ids[2]
        breakpoint_4 = breakpoint_ids[3]
        breakpoint_5 = breakpoint_ids[4]
        breakpoint_6 = breakpoint_ids[5]
        breakpoint_7 = breakpoint_ids[6]
        breakpoint_8 = breakpoint_ids[7]
        self.continue_to_breakpoint(breakpoint_1)

        # Expressions at breakpoint 1, which is in main
        self.assertEvaluate("var1", "20")
        # Empty expression should equate to the previous expression.
        if context == "repl":
            self.assertEvaluate("", "20")
        else:
            self.assertEvaluateFailure("")
        self.assertEvaluate("var2", "21")
        if context == "repl":
            self.assertEvaluate("", "21")
            self.assertEvaluate("", "21")
        self.assertEvaluate("static_int", "42")
        self.assertEvaluate("non_static_int", "43")
        self.assertEvaluate("struct1.foo", "15")
        self.assertEvaluate("struct2->foo", "16")

        if self.isResultExpandedDescription():
            self.assertEvaluate(
                "struct1",
                r"\(my_struct\) (struct1|\$\d+) = \(foo = 15\)",
            )
            self.assertEvaluate("struct2", r"\(my_struct \*\) (struct2|\$\d+) = 0x.*")
            self.assertEvaluate(
                "struct3", r"\(my_struct \*\) (struct3|\$\d+) = nullptr"
            )
        else:
            self.assertEvaluate(
                "struct1",
                (
                    re.escape("{foo:15}")
                    if enableAutoVariableSummaries
                    else "my_struct @ 0x"
                ),
            )
            self.assertEvaluate(
                "struct2", "0x.* {foo:16}" if enableAutoVariableSummaries else "0x.*"
            )
            self.assertEvaluate("struct3", "0x.*0")

        if context == "repl":
            # In the repl context expressions may be interpreted as lldb
            # commands since no variables have the same name as the command.
            self.assertEvaluate("list", r".*")
        else:
            self.assertEvaluateFailure("list")  # local variable of a_function

        self.assertEvaluateFailure("my_struct")  # type name
        self.assertEvaluateFailure("int")  # type name
        self.assertEvaluateFailure("foo")  # member of my_struct

        if self.isExpressionParsedExpected():
            self.assertEvaluate("a_function", "0x.*a.out`a_function.*")
            self.assertEvaluate("a_function(1)", "1")
            self.assertEvaluate("var2 + struct1.foo", "36")
            self.assertEvaluate("foo_func", "0x.*a.out`foo_func.*")
            self.assertEvaluate("foo_var", "44")
        else:
            self.assertEvaluateFailure("a_function")
            self.assertEvaluateFailure("a_function(1)")
            self.assertEvaluateFailure("var2 + struct1.foo")
            self.assertEvaluateFailure("foo_func")
            self.assertEvaluateFailure("foo_var")

        # Expressions at breakpoint 2, which is an anonymous block
        self.continue_to_breakpoint(breakpoint_2)
        self.assertEvaluate("var1", "20")
        self.assertEvaluate("var2", "2")  # different variable with the same name
        self.assertEvaluate("static_int", "42")
        self.assertEvaluate(
            "non_static_int", "10"
        )  # different variable with the same name
        if self.isResultExpandedDescription():
            self.assertEvaluate(
                "struct1",
                r"\(my_struct\) (struct1|\$\d+) = \(foo = 15\)",
            )
        else:
            self.assertEvaluate(
                "struct1",
                (
                    re.escape("{foo:15}")
                    if enableAutoVariableSummaries
                    else "my_struct @ 0x"
                ),
            )
        self.assertEvaluate("struct1.foo", "15")
        self.assertEvaluate("struct2->foo", "16")

        if self.isExpressionParsedExpected():
            self.assertEvaluate("a_function", "0x.*a.out`a_function.*")
            self.assertEvaluate("a_function(1)", "1")
            self.assertEvaluate("var2 + struct1.foo", "17")
            self.assertEvaluate("foo_func", "0x.*a.out`foo_func.*")
            self.assertEvaluate("foo_var", "44")
        else:
            self.assertEvaluateFailure("a_function")
            self.assertEvaluateFailure("a_function(1)")
            self.assertEvaluateFailure("var2 + struct1.foo")
            self.assertEvaluateFailure("foo_func")
            self.assertEvaluateFailure("foo_var")

        # Expressions at breakpoint 3, which is inside a_function
        self.continue_to_breakpoint(breakpoint_3)
        self.assertEvaluate("list", "42")
        self.assertEvaluate("static_int", "42")
        self.assertEvaluate("non_static_int", "43")

        self.assertEvaluateFailure("var1")
        self.assertEvaluateFailure("var2")
        self.assertEvaluateFailure("struct1")
        self.assertEvaluateFailure("struct1.foo")
        self.assertEvaluateFailure("struct2->foo")
        self.assertEvaluateFailure("var2 + struct1.foo")

        if self.isExpressionParsedExpected():
            self.assertEvaluate("a_function", "0x.*a.out`a_function.*")
            self.assertEvaluate("a_function(1)", "1")
            self.assertEvaluate("list + 1", "43")
            self.assertEvaluate("foo_func", "0x.*a.out`foo_func.*")
            self.assertEvaluate("foo_var", "44")
        else:
            self.assertEvaluateFailure("a_function")
            self.assertEvaluateFailure("a_function(1)")
            self.assertEvaluateFailure("list + 1")
            self.assertEvaluateFailure("foo_func")
            self.assertEvaluateFailure("foo_var")

        # Now we check that values are updated after stepping
        self.continue_to_breakpoint(breakpoint_4)
        self.assertEvaluate("my_vec", "size=2")
        self.continue_to_breakpoint(breakpoint_5)
        self.assertEvaluate("my_vec", "size=3")

        self.assertEvaluate("my_map", "size=2")
        self.continue_to_breakpoint(breakpoint_6)
        self.assertEvaluate("my_map", "size=3")

        self.assertEvaluate("my_bool_vec", "size=1")
        self.continue_to_breakpoint(breakpoint_7)
        self.assertEvaluate("my_bool_vec", "size=2")

        self.continue_to_breakpoint(breakpoint_8)
        # Test memory read, especially with 'empty' repeat commands.
        if context == "repl":
            self.assertEvaluate("memory read -c 1 &my_ints", ".* 05 .*\n")
            self.assertEvaluate("", ".* 0a .*\n")
            self.assertEvaluate("", ".* 0f .*\n")
            self.assertEvaluate("", ".* 14 .*\n")
            self.assertEvaluate("", ".* 19 .*\n")

        self.continue_to_exit()

    @skipIfWindows
    def test_generic_evaluate_expressions(self):
        # Tests context-less expression evaluations
        self.run_test_evaluate_expressions(enableAutoVariableSummaries=False)

    @skipIfWindows
    def test_repl_evaluate_expressions(self):
        # Tests expression evaluations that are triggered from the Debug Console
        self.run_test_evaluate_expressions("repl", enableAutoVariableSummaries=False)

    @skipIfWindows
    def test_watch_evaluate_expressions(self):
        # Tests expression evaluations that are triggered from a watch expression
        self.run_test_evaluate_expressions("watch", enableAutoVariableSummaries=True)

    @skipIfWindows
    def test_hover_evaluate_expressions(self):
        # Tests expression evaluations that are triggered when hovering on the editor
        self.run_test_evaluate_expressions("hover", enableAutoVariableSummaries=False)

    @skipIfWindows
    def test_variable_evaluate_expressions(self):
        # Tests expression evaluations that are triggered in the variable explorer
        self.run_test_evaluate_expressions("variable", enableAutoVariableSummaries=True)
