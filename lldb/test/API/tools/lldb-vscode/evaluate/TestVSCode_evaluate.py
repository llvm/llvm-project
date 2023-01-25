"""
Test lldb-vscode completions request
"""


import lldbvscode_testcase
import vscode
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class TestVSCode_variables(lldbvscode_testcase.VSCodeTestCaseBase):

    def assertEvaluate(self, expression, regex):
        self.assertRegexpMatches(
            self.vscode.request_evaluate(expression, context=self.context)['body']['result'],
            regex)

    def assertEvaluateFailure(self, expression):
        self.assertNotIn('result',
            self.vscode.request_evaluate(expression, context=self.context)['body'])

    def isExpressionParsedExpected(self):
        return self.context != "hover"

    def run_test_evaluate_expressions(self, context=None):
        """
            Tests the evaluate expression request at different breakpoints
        """
        self.context = context
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        self.set_source_breakpoints(
            source,
            [
                line_number(source, "// breakpoint 1"),
                line_number(source, "// breakpoint 2"),
                line_number(source, "// breakpoint 3"),
                line_number(source, "// breakpoint 4"),
                line_number(source, "// breakpoint 5"),
                line_number(source, "// breakpoint 6"),
                line_number(source, "// breakpoint 7"),
            ]
        )
        self.continue_to_next_stop()

        # Expressions at breakpoint 1, which is in main
        self.assertEvaluate("var1", "20")
        self.assertEvaluate("var2", "21")
        self.assertEvaluate("static_int", "42")
        self.assertEvaluate("non_static_int", "43")
        self.assertEvaluate("struct1", "my_struct @ 0x.*")
        self.assertEvaluate("struct1.foo", "15")
        self.assertEvaluate("struct2->foo", "16")

        self.assertEvaluateFailure("var") # local variable of a_function
        self.assertEvaluateFailure("my_struct") # type name
        self.assertEvaluateFailure("int") # type name
        self.assertEvaluateFailure("foo") # member of my_struct

        if self.isExpressionParsedExpected():
            self.assertEvaluate(
                "a_function",
                "0x.*a.out`a_function.*")
            self.assertEvaluate("a_function(1)", "1")
            self.assertEvaluate("var2 + struct1.foo", "36")
            self.assertEvaluate(
                "foo_func",
                "0x.*a.out`foo_func.*")
            self.assertEvaluate("foo_var", "44")
        else:
            self.assertEvaluateFailure("a_function")
            self.assertEvaluateFailure("a_function(1)")
            self.assertEvaluateFailure("var2 + struct1.foo")
            self.assertEvaluateFailure("foo_func")
            self.assertEvaluateFailure("foo_var")

        # Expressions at breakpoint 2, which is an anonymous block
        self.continue_to_next_stop()
        self.assertEvaluate("var1", "20")
        self.assertEvaluate("var2", "2") # different variable with the same name
        self.assertEvaluate("static_int", "42")
        self.assertEvaluate("non_static_int", "10") # different variable with the same name
        self.assertEvaluate("struct1", "my_struct @ 0x.*")
        self.assertEvaluate("struct1.foo", "15")
        self.assertEvaluate("struct2->foo", "16")

        if self.isExpressionParsedExpected():
            self.assertEvaluate(
                "a_function",
                "0x.*a.out`a_function.*")
            self.assertEvaluate("a_function(1)", "1")
            self.assertEvaluate("var2 + struct1.foo", "17")
            self.assertEvaluate(
                "foo_func",
                "0x.*a.out`foo_func.*")
            self.assertEvaluate("foo_var", "44")
        else:
            self.assertEvaluateFailure("a_function")
            self.assertEvaluateFailure("a_function(1)")
            self.assertEvaluateFailure("var2 + struct1.foo")
            self.assertEvaluateFailure("foo_func")
            self.assertEvaluateFailure("foo_var")

        # Expressions at breakpoint 3, which is inside a_function
        self.continue_to_next_stop()
        self.assertEvaluate("var", "42")
        self.assertEvaluate("static_int", "42")
        self.assertEvaluate("non_static_int", "43")

        self.assertEvaluateFailure("var1")
        self.assertEvaluateFailure("var2")
        self.assertEvaluateFailure("struct1")
        self.assertEvaluateFailure("struct1.foo")
        self.assertEvaluateFailure("struct2->foo")
        self.assertEvaluateFailure("var2 + struct1.foo")

        if self.isExpressionParsedExpected():
            self.assertEvaluate(
                "a_function",
                "0x.*a.out`a_function.*")
            self.assertEvaluate("a_function(1)", "1")
            self.assertEvaluate("var + 1", "43")
            self.assertEvaluate(
                "foo_func",
                "0x.*a.out`foo_func.*")
            self.assertEvaluate("foo_var", "44")
        else:
            self.assertEvaluateFailure("a_function")
            self.assertEvaluateFailure("a_function(1)")
            self.assertEvaluateFailure("var + 1")
            self.assertEvaluateFailure("foo_func")
            self.assertEvaluateFailure("foo_var")

        # Now we check that values are updated after stepping
        self.continue_to_next_stop()
        self.assertEvaluate("my_vec", "size=2")
        self.continue_to_next_stop()
        self.assertEvaluate("my_vec", "size=3")

        self.assertEvaluate("my_map", "size=2")
        self.continue_to_next_stop()
        self.assertEvaluate("my_map", "size=3")
        
        self.assertEvaluate("my_bool_vec", "size=1")
        self.continue_to_next_stop()
        self.assertEvaluate("my_bool_vec", "size=2")

    @skipIfWindows
    @skipIfRemote
    def test_generic_evaluate_expressions(self):
        # Tests context-less expression evaluations
        self.run_test_evaluate_expressions()

    @skipIfWindows
    @skipIfRemote
    def test_repl_evaluate_expressions(self):
        # Tests expression evaluations that are triggered from the Debug Console
        self.run_test_evaluate_expressions("repl")

    @skipIfWindows
    @skipIfRemote
    def test_watch_evaluate_expressions(self):
        # Tests expression evaluations that are triggered from a watch expression
        self.run_test_evaluate_expressions("watch")

    @skipIfWindows
    @skipIfRemote
    def test_hover_evaluate_expressions(self):
        # Tests expression evaluations that are triggered when hovering on the editor
        self.run_test_evaluate_expressions("hover")
