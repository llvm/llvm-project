from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentManyBreakpoints(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @skipIfOutOfTreeDebugserver
    @expectedFailureAll(
        archs=["aarch64"], oslist=["freebsd"], bugnumber="llvm.org/pr49433"
    )
    def test(self):
        """Test 100 breakpoints from 100 threads."""
        self.build()
        self.do_thread_actions(num_breakpoint_threads=100)
