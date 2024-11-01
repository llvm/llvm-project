"""
Ensure that when the interrupt is raised we still make frame 0.
and make sure "GetNumFrames" isn't interrupted.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestInterruptingBacktrace(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(oslist=["linux"], archs=["arm"])
    def test_backtrace_interrupt(self):
        """Use RequestInterrupt followed by stack operations
        to ensure correct interrupt behavior for stacks."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.bt_interrupt_test()

    def bt_interrupt_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        # Now continue, and when we stop we will have crashed.
        process.Continue()
        self.dbg.RequestInterrupt()

        # Be sure to turn this off again:
        def cleanup():
            if self.dbg.InterruptRequested():
                self.dbg.CancelInterruptRequest()

        self.addTearDownHook(cleanup)

        frame_0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame_0.IsValid(), "Got a good 0th frame")
        # The interrupt flag is up already, so any attempt to backtrace
        # should be cut short:
        frame_1 = thread.GetFrameAtIndex(1)
        self.assertFalse(frame_1.IsValid(), "Prevented from getting more frames")
        # Since GetNumFrames is a contract, we don't interrupt it:
        num_frames = thread.GetNumFrames()
        print(f"Number of frames: {num_frames}")
        self.assertGreater(num_frames, 1, "Got many frames")

        self.dbg.CancelInterruptRequest()
