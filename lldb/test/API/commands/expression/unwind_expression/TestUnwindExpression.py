"""
Test stopping at a breakpoint in an expression, and unwinding from there.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class UnwindFromExpressionTest(TestBase):
    main_spec = lldb.SBFileSpec("main.cpp", False)

    def build_and_run_to_bkpt(self):
        self.build()

        (target, process, self.thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// Set a breakpoint here to get started", self.main_spec
        )

        # Next set a breakpoint in this function, set up Expression options to stop on
        # breakpoint hits, and call the function.
        self.fun_bkpt = self.target().BreakpointCreateBySourceRegex(
            "// Stop inside the function here.", self.main_spec
        )
        self.assertTrue(self.fun_bkpt, VALID_BREAKPOINT)

    @no_debug_info_test
    @expectedFailureAll(bugnumber="llvm.org/pr33164")
    def test_conditional_bktp(self):
        """
        Test conditional breakpoint handling in the IgnoreBreakpoints = False case
        """
        self.build_and_run_to_bkpt()

        self.fun_bkpt.SetCondition("0")  # Should not get hit
        options = lldb.SBExpressionOptions()
        options.SetIgnoreBreakpoints(False)
        options.SetUnwindOnError(False)

        main_frame = self.thread.GetFrameAtIndex(0)
        val = main_frame.EvaluateExpression("second_function(47)", options)
        self.assertSuccess(val.GetError(), "We did complete the execution.")
        self.assertEqual(47, val.GetValueAsSigned())

    @add_test_categories(["pyapi"])
    @expectedFlakeyNetBSD
    def test_unwind_expression(self):
        """Test unwinding from an expression."""
        self.build_and_run_to_bkpt()

        # Run test with varying one thread timeouts to also test the halting
        # logic in the IgnoreBreakpoints = False case
        self.do_unwind_test(self.thread, self.fun_bkpt, 1000)
        self.do_unwind_test(self.thread, self.fun_bkpt, 100000)

    def do_unwind_test(self, thread, bkpt, timeout):
        #
        # Use Python API to evaluate expressions while stopped in a stack frame.
        #
        main_frame = thread.GetFrameAtIndex(0)

        options = lldb.SBExpressionOptions()
        options.SetIgnoreBreakpoints(False)
        options.SetUnwindOnError(False)
        options.SetOneThreadTimeoutInMicroSeconds(timeout)

        val = main_frame.EvaluateExpression("a_function_to_call()", options)

        self.assertTrue(val.GetError().Fail(), "We did not complete the execution.")
        error_str = val.GetError().GetCString()
        self.assertTrue(
            "Execution was interrupted, reason: breakpoint" in error_str,
            "And the reason was right.",
        )

        thread = lldbutil.get_one_thread_stopped_at_breakpoint(self.process(), bkpt)
        self.assertTrue(thread.IsValid(), "We are indeed stopped at our breakpoint")

        # Now unwind the expression, and make sure we got back to where we
        # started.
        self.assertSuccess(
            thread.UnwindInnermostExpression(), "We succeeded in unwinding"
        )

        cur_frame = thread.GetFrameAtIndex(0)
        self.assertTrue(cur_frame.IsEqual(main_frame), "We got back to the main frame.")
