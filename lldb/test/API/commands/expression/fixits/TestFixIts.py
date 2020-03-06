"""
Test calling an expression with errors that a FixIt can fix.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandWithFixits(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @skipUnlessDarwin
    def test_with_target(self):
        """Test calling expressions with errors that can be fixed by the FixIts."""
        self.build()
        self.try_expressions()

    def test_with_dummy_target(self):
        """Test calling expressions in the dummy target with errors that can be fixed by the FixIts."""
        ret_val = lldb.SBCommandReturnObject()
        result = self.dbg.GetCommandInterpreter().HandleCommand("expression ((1 << 16) - 1))", ret_val)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "The expression was successful.")
        self.assertTrue("Fix-it applied" in ret_val.GetError(), "Found the applied FixIt.")

    def try_expressions(self):
        """Test calling expressions with errors that can be fixed by the FixIts."""
        (target, process, self.thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                        'Stop here to evaluate expressions', self.main_source_spec)

        options = lldb.SBExpressionOptions()
        options.SetAutoApplyFixIts(True)

        frame = self.thread.GetFrameAtIndex(0)

        # Try with one error:
        value = frame.EvaluateExpression("my_pointer.first", options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertTrue(value.GetValueAsUnsigned() == 10)

        # Try with two errors:
        two_error_expression = "my_pointer.second->a"
        value = frame.EvaluateExpression(two_error_expression, options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertTrue(value.GetValueAsUnsigned() == 20)

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
