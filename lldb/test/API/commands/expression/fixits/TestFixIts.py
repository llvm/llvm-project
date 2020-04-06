"""
Test calling an expression with errors that a FixIt can fix.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandWithFixits(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_dummy_target(self):
        """Test calling expressions in the dummy target with errors that can be fixed by the FixIts."""

        # Enable fix-its as they were intentionally disabled by TestBase.setUp.
        self.runCmd("settings set target.auto-apply-fixits true")

        ret_val = lldb.SBCommandReturnObject()
        result = self.dbg.GetCommandInterpreter().HandleCommand("expression ((1 << 16) - 1))", ret_val)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "The expression was successful.")
        self.assertTrue("Fix-it applied" in ret_val.GetError(), "Found the applied FixIt.")

    def test_with_target(self):
        """Test calling expressions with errors that can be fixed by the FixIts."""
        self.build()
        (target, process, self.thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                        'Stop here to evaluate expressions',
                                         lldb.SBFileSpec("main.cpp"))

        options = lldb.SBExpressionOptions()
        options.SetAutoApplyFixIts(True)

        top_level_options = lldb.SBExpressionOptions()
        top_level_options.SetAutoApplyFixIts(True)
        top_level_options.SetTopLevel(True)

        frame = self.thread.GetFrameAtIndex(0)

        # Try with one error:
        value = frame.EvaluateExpression("my_pointer.first", options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEquals(value.GetValueAsUnsigned(), 10)

        # Try with one error in a top-level expression.
        # The Fix-It changes "ptr.m" to "ptr->m".
        expr = "struct X { int m; }; X x; X *ptr = &x; int m = ptr.m;"
        value = frame.EvaluateExpression(expr, top_level_options)
        # A successfully parsed top-level expression will yield an error
        # that there is 'no value'. If a parsing error would have happened we
        # would get a different error kind, so let's check the error kind here.
        self.assertEquals(value.GetError().GetCString(), "error: No value")

        # Try with two errors:
        two_error_expression = "my_pointer.second->a"
        value = frame.EvaluateExpression(two_error_expression, options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEquals(value.GetValueAsUnsigned(), 20)

        # Try a Fix-It that is stored in the 'note:' diagnostic of an error.
        # The Fix-It here is adding parantheses around the ToStr parameters.
        fixit_in_note_expr ="#define ToStr(x) #x\nToStr(0 {, })"
        value = frame.EvaluateExpression(fixit_in_note_expr, options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success(), value.GetError())
        self.assertEquals(value.GetSummary(), '"(0 {, })"')

        # Now turn off the fixits, and the expression should fail:
        options.SetAutoApplyFixIts(False)
        value = frame.EvaluateExpression(two_error_expression, options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Fail())
        error_string = value.GetError().GetCString()
        self.assertTrue(
            error_string.find("fixed expression suggested:") != -1,
            "Fix was suggested")
        self.assertTrue(
            error_string.find("my_pointer->second.a") != -1,
            "Fix was right")
