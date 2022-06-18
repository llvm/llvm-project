"""
Tests target.expr-error-limit.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @no_debug_info_test
    def test(self):
        # FIXME: The only reason this test needs to create a real target is because
        # the settings of the dummy target can't be changed with `settings set`.
        self.build()
        target = self.createTestTarget()

        # Our test expression that is just several lines of malformed
        # integer literals (with a 'yerror' integer suffix). Every error
        # has its own unique string (1, 2, 3, 4) and is on its own line
        # that we can later find it when Clang prints the respective source
        # code for each error to the error output.
        # For example, in the error output below we would look for the
        # unique `1yerror` string:
        #     error: <expr>:1:2: invalid suffix 'yerror' on integer constant
        #     1yerror
        #     ^
        expr = "1yerror;\n2yerror;\n3yerror;\n4yerror;"

        options = lldb.SBExpressionOptions()
        options.SetAutoApplyFixIts(False)

        # Evaluate the expression and check that only the first 2 errors are
        # emitted.
        self.runCmd("settings set target.expr-error-limit 2")
        eval_result = target.EvaluateExpression(expr, options)
        self.assertIn("1yerror", str(eval_result.GetError()))
        self.assertIn("2yerror", str(eval_result.GetError()))
        self.assertNotIn("3yerror", str(eval_result.GetError()))
        self.assertNotIn("4yerror", str(eval_result.GetError()))

        # Change to a 3 errors and check again which errors are emitted.
        self.runCmd("settings set target.expr-error-limit 3")
        eval_result = target.EvaluateExpression(expr, options)
        self.assertIn("1yerror", str(eval_result.GetError()))
        self.assertIn("2yerror", str(eval_result.GetError()))
        self.assertIn("3yerror", str(eval_result.GetError()))
        self.assertNotIn("4yerror", str(eval_result.GetError()))

        # Disable the error limit and make sure all errors are emitted.
        self.runCmd("settings set target.expr-error-limit 0")
        eval_result = target.EvaluateExpression(expr, options)
        self.assertIn("1yerror", str(eval_result.GetError()))
        self.assertIn("2yerror", str(eval_result.GetError()))
        self.assertIn("3yerror", str(eval_result.GetError()))
        self.assertIn("4yerror", str(eval_result.GetError()))
