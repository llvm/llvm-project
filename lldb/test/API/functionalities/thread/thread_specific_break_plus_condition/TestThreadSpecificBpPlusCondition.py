"""
Test that we obey thread conditioned breakpoints and expression
conditioned breakpoints simultaneously
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireThreadSupport
class ThreadSpecificBreakPlusConditionTestCase(TestBase):
    # test frequently times out or hangs
    @skipIfDarwin
    # hits break in another thread in testrun
    @add_test_categories(["pyapi"])
    @expectedFlakeyNetBSD
    @skipIfWindows  # This test is flaky on Windows
    def test_python(self):
        """Test that we obey thread conditioned breakpoints."""
        self.build()

        # Set a breakpoint in the thread body, and make it active for only the
        # first thread.  Several threads can run into it at once.
        (
            _,
            process,
            victim_thread,
            break_thread_body,
        ) = lldbutil.run_to_source_breakpoint(
            self,
            "Break here in thread body.",
            lldb.SBFileSpec("main.cpp"),
            only_one_thread=False,
        )

        # Pick one of the threads, and change the breakpoint so it ONLY stops for this thread,
        # but add a condition that it won't stop for this thread's my_value.  The other threads
        # pass the condition, so they should stop, but if the thread-specification is working
        # they should not stop.  So nobody should hit the breakpoint anymore, and we should
        # just exit cleanly.

        frame = victim_thread.GetFrameAtIndex(0)
        value = frame.FindVariable("my_value").GetValueAsSigned(0)
        self.assertTrue(
            value > 0 and value < 11, "Got a reasonable value for my_value."
        )

        cond_string = "my_value != %d" % (value)

        break_thread_body.SetThreadID(victim_thread.GetThreadID())
        break_thread_body.SetCondition(cond_string)

        process.Continue()

        next_stop_state = process.GetState()
        self.assertEqual(
            next_stop_state,
            lldb.eStateExited,
            "We should have not hit the breakpoint again.",
        )
