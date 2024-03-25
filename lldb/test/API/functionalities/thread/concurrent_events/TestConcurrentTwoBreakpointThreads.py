from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentTwoBreakpointThreads(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @expectedFailureAll(
        archs=["aarch64"], oslist=["freebsd"], bugnumber="llvm.org/pr49433"
    )
    def test(self):
        """Test two threads that trigger a breakpoint."""
        self.build()
        self.do_thread_actions(num_breakpoint_threads=2)
