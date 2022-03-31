"""
Make sure the stop reason of a thread that did not run
during an expression is not changed by running the expression
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestStopReasonAfterExpression(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows
    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr48415")
    @expectedFlakeyNetBSD
    def test_thread_state_after_expr(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.do_test()

    def do_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
            "Set a breakpoint here", self.main_source_file)

        self.assertEqual(bkpt.GetNumLocations(), 2, "Got two locations")

        # So now thread holds the main thread.  Continue to hit the
        # breakpoint again on the spawned thread:

        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Hit the breakpoint the second time")
        other_thread = threads[0]

        self.assertNotEqual(thread.GetThreadID(), other_thread.GetThreadID(),
                            "A different thread")
        # Run an expression ONLY on other_thread.  Don't let thread run:
        options = lldb.SBExpressionOptions()
        options.SetTryAllThreads(False)
        options.SetStopOthers(True)

        result = thread.frames[0].EvaluateExpression('(int) printf("Hello\\n")', options)
        self.assertSuccess(result.GetError(), "Expression failed")

        stop_reason = other_thread.GetStopReason()

        self.assertEqual(stop_reason, lldb.eStopReasonBreakpoint,
                         "Still records stopped at breakpoint: %s"
                         %(lldbutil.stop_reason_to_str(stop_reason)))
        self.assertEqual(other_thread.GetStopReasonDataAtIndex(0), 1,
                         "Still records stopped at right breakpoint")

