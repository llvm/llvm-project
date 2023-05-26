"""
Test that we work properly with classes with the trivial_abi attribute
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTrivialABI(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessSupportedTypeAttribute("trivial_abi")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37995")
    @expectedFailureAll(
        archs=["aarch64"], oslist=["freebsd", "linux"], bugnumber="llvm.org/pr44161"
    )
    def test_call_trivial(self):
        """Test that we can print a variable & call a function with a trivial ABI class."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.expr_test(True)

    @skipUnlessSupportedTypeAttribute("trivial_abi")
    # fixed for SysV-x86_64 ABI, but not Windows-x86_64
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr36870")
    @expectedFailureAll(
        archs=["arm", "aarch64"],
        oslist=["freebsd", "linux"],
        bugnumber="llvm.org/pr44161",
    )
    @expectedFailureAll(
        archs=["arm64", "arm64e"], bugnumber="<rdar://problem/57844240>"
    )
    def test_call_nontrivial(self):
        """Test that we can print a variable & call a function on the same class w/o the trivial ABI marker."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.expr_test(False)

    def check_value(self, test_var, ivar_value):
        self.assertSuccess(test_var.GetError(), "Invalid valobj")
        ivar = test_var.GetChildMemberWithName("ivar")
        self.assertSuccess(test_var.GetError(), "Failed to fetch ivar")
        self.assertEqual(
            ivar_value, ivar.GetValueAsSigned(), "Got the right value for ivar"
        )

    def check_frame(self, thread):
        frame = thread.frames[0]
        inVal_var = frame.FindVariable("inVal")
        self.check_value(inVal_var, 10)

        options = lldb.SBExpressionOptions()
        inVal_expr = frame.EvaluateExpression("inVal", options)
        self.check_value(inVal_expr, 10)

        thread.StepOut()
        outVal_ret = thread.GetStopReturnValue()
        self.check_value(outVal_ret, 30)

    def expr_test(self, trivial):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        # Stop in a function that takes a trivial value, and try both frame var & expr to get its value:
        if trivial:
            self.check_frame(thread)
            return

        # Now continue to the same thing without the trivial_abi and see if we get that right:
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Hit my breakpoint the second time.")

        self.check_frame(threads[0])
