"""
Test SB API support for identifying artificial (tail call) frames.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestArtificialFrameThreadStepOut1(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def prepare_thread(self):
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        # Here's what we expect to see in the backtrace:
        #   frame #0: ... a.out`sink() at main.cpp:13:4 [opt]
        #   frame #1: ... a.out`func3() at main.cpp:14:1 [opt] [artificial]
        #   frame #2: ... a.out`func2() at main.cpp:18:62 [opt]
        #   frame #3: ... a.out`func1() at main.cpp:18:85 [opt] [artificial]
        #   frame #4: ... a.out`main at main.cpp:23:3 [opt]
        return thread

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr26265")
    def test_stepping_out_past_artificial_frame(self):
        self.build()
        thread = self.prepare_thread()

        # Frame #0's ancestor is artificial. Stepping out should move to
        # frame #2, because we behave as-if artificial frames were not present.
        thread.StepOut()
        frame2 = thread.GetSelectedFrame()
        self.assertEqual(frame2.GetDisplayFunctionName(), "func2()")
        self.assertFalse(frame2.IsArtificial())

        # Ditto: stepping out of frame #2 should move to frame #4.
        thread.StepOut()
        frame4 = thread.GetSelectedFrame()
        self.assertEqual(frame4.GetDisplayFunctionName(), "main")
        self.assertFalse(frame2.IsArtificial())

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr26265")
    def test_return_past_artificial_frame(self):
        self.build()
        thread = self.prepare_thread()

        value = lldb.SBValue()

        # Frame #0's ancestor is artificial. Returning from frame #0 should move
        # to frame #2.
        thread.ReturnFromFrame(thread.GetSelectedFrame(), value)
        frame2 = thread.GetSelectedFrame()
        self.assertEqual(frame2.GetDisplayFunctionName(), "func2()")
        self.assertFalse(frame2.IsArtificial())

        # Ditto: stepping out of frame #2 should move to frame #4.
        thread.ReturnFromFrame(thread.GetSelectedFrame(), value)
        frame4 = thread.GetSelectedFrame()
        self.assertEqual(frame4.GetDisplayFunctionName(), "main")
        self.assertFalse(frame2.IsArtificial())
