"""
Test that --allow-jit=false does disallow JITting:
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestAllowJIT(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_allow_jit_expr_command(self):
        """Test the --allow-jit command line flag"""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.expr_cmd_test()

    def test_allow_jit_options(self):
        """Test the SetAllowJIT SBExpressionOption setting"""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.expr_options_test()

    def expr_cmd_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        frame = thread.GetFrameAtIndex(0)

        # First make sure we can call the function with 
        interp = self.dbg.GetCommandInterpreter()
        self.expect("expr --allow-jit 1 -- call_me(10)",
                    substrs = ["(int) $", "= 18"])
        # Now make sure it fails with the "can't IR interpret message" if allow-jit is false:
        self.expect("expr --allow-jit 0 -- call_me(10)",
                    error=True,
                    substrs = ["Can't evaluate the expression without a running target"])

    def expr_options_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        frame = thread.GetFrameAtIndex(0)

        # First make sure we can call the function with the default option set. 
        options = lldb.SBExpressionOptions()
        # Check that the default is to allow JIT:
        self.assertEqual(options.GetAllowJIT(), True, "Default is true")

        # Now use the options:
        result = frame.EvaluateExpression("call_me(10)", options)
        self.assertSuccess(result.GetError())
        self.assertEqual(result.GetValueAsSigned(), 18, "got the right value.")

        # Now disallow JIT and make sure it fails:
        options.SetAllowJIT(False)
        # Check that we got the right value:
        self.assertEqual(options.GetAllowJIT(), False, "Got False after setting to False")

        # Again use it and ensure we fail:
        result = frame.EvaluateExpression("call_me(10)", options)
        self.assertTrue(result.GetError().Fail(), "expression failed with no JIT")
        self.assertTrue("Can't evaluate the expression without a running target" in result.GetError().GetCString(), "Got right error")

        # Finally set the allow JIT value back to true and make sure that works:
        options.SetAllowJIT(True)
        self.assertEqual(options.GetAllowJIT(), True, "Set back to True correctly")

        # And again, make sure this works:
        result = frame.EvaluateExpression("call_me(10)", options)
        self.assertSuccess(result.GetError())
        self.assertEqual(result.GetValueAsSigned(), 18, "got the right value.")

