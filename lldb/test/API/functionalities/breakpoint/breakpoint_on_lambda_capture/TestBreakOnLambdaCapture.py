"""
Test that if we hit a breakpoint on a lambda capture
on two threads at the same time we stop only for
the correct one.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestBreakOnLambdaCapture(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def test_break_on_lambda_capture(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")

        (target, process, main_thread, _) = lldbutil.run_to_source_breakpoint(self,
                                                "First break", self.main_source_file)

        # FIXME: This is working around a separate bug. If you hit a breakpoint and
        # run an expression and it is the first expression you've ever run, on
        # Darwin that will involve running the ObjC runtime parsing code, and we'll
        # be in the middle of that when we do PerformAction on the other thread,
        # which will cause the condition expression to fail.  Calling another
        # expression first works around this.
        val_obj = main_thread.frame[0].EvaluateExpression("true")
        self.assertSuccess(val_obj.GetError(), "Ran our expression successfully")
        self.assertEqual(val_obj.value, "true", "Value was true.")

        bkpt = target.BreakpointCreateBySourceRegex("Break here in the helper",
                                                    self.main_source_file);

        bkpt.SetCondition("enable && usec == 1")
        process.Continue()

        # This is hard to test definitively, becuase it requires hitting
        # a breakpoint on multiple threads at the same time.  On Darwin, this
        # will happen pretty much ever time we continue.  What we are really
        # asserting is that we only ever stop on one thread, so we approximate that
        # by continuing 20 times and assert we only ever hit the first thread.  Either
        # this is a platform that only reports one hit at a time, in which case all
        # this code is unused, or we actually didn't hit the other thread.

        for idx in range(0, 20):
            process.Continue()
            for thread in process.threads:
                if thread.id == main_thread.id:
                    self.assertStopReason(thread.stop_reason, lldb.eStopReasonBreakpoint)
                else:
                    self.assertStopReason(thread.stop_reason, lldb.eStopReasonNone)
