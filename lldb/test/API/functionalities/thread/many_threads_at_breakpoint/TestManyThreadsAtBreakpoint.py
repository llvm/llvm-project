"""
Test that each of many threads stopped at one breakpoint gets its own stop
info, PC and registers.

The threads reach the breakpoint together, so one stop can report many of them.
Each thread first stores its own thread ID in a local, so a thread given
another thread's PC or registers shows the wrong ID in its frame.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


@requireThreadSupport
class ManyThreadsAtBreakpointTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        """Test that every thread stops at the breakpoint once, in its own frame."""
        self.build()
        _, process, _, bkpt = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp"), only_one_thread=False
        )
        line = line_number("main.cpp", "// break here")

        indexes = []
        stopped = lldbutil.get_threads_stopped_at_breakpoint(process, bkpt)
        while stopped:
            for thread in stopped:
                frame = thread.GetFrameAtIndex(0)
                self.assertEqual(frame.GetLineEntry().GetLine(), line)
                self.assertEqual(
                    frame.FindVariable("tid").GetValueAsUnsigned(), thread.GetThreadID()
                )
                indexes.append(frame.FindVariable("index").GetValueAsSigned())
            stopped = lldbutil.continue_to_breakpoint(process, bkpt)

        # main.cpp starts 32 threads.
        self.assertEqual(sorted(indexes), list(range(32)))
        self.assertState(process.GetState(), lldb.eStateExited)
