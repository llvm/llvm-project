"""
Test that if we hit a breakpoint on two threads at the 
same time, one of which passes the condition, one not,
we only have a breakpoint stop reason for the one that
passed the condition.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestTwoHitsOneActual(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_two_hits_one_actual(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.sample_test()

    def sample_test(self):
        """You might use the test implementation in several ways, say so here."""

        (target, process, main_thread, _) = lldbutil.run_to_source_breakpoint(self,
                                   "Set bkpt here to get started", self.main_source_file)
        # This is working around a separate bug.  If you hit a breakpoint and
        # run an expression and it is the first expression you've ever run, on
        # Darwin that will involve running the ObjC runtime parsing code, and we'll
        # be in the middle of that when we do PerformAction on the other thread,
        # which will cause the condition expression to fail.  Calling another
        # expression first works around this.
        val_obj = main_thread.frame[0].EvaluateExpression("main_usec==1")
        self.assertSuccess(val_obj.GetError(), "Ran our expression successfully")
        self.assertEqual(val_obj.value, "true", "Value was true.")
        # Set two breakpoints just to test the multiple location logic:
        bkpt1 = target.BreakpointCreateBySourceRegex("Break here in the helper", self.main_source_file);
        bkpt2 = target.BreakpointCreateBySourceRegex("Break here in the helper", self.main_source_file);

        # This one will never be hit:
        bkpt1.SetCondition("usec == 100")
        # This one will only be hit on the main thread:
        bkpt2.SetCondition("usec == 1")

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
                    self.assertEqual(thread.stop_reason, lldb.eStopReasonBreakpoint)
                else:
                    self.assertEqual(thread.stop_reason, lldb.eStopReasonNone)

                
