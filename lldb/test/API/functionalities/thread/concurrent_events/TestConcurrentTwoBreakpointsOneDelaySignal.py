from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentTwoBreakpointsOneDelaySignal(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @expectedFlakeyNetBSD
    @expectedFailureAll(
        archs=["aarch64"], oslist=["freebsd"], bugnumber="llvm.org/pr49433"
    )
    def test(self):
        """Test two threads that trigger a breakpoint and one (1 second delay) signal thread."""
        self.build()
        self.do_thread_actions(num_breakpoint_threads=2, num_delay_signal_threads=1)
