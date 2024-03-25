from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentSignalWatchBreak(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @expectedFailureNetBSD
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test a signal/watchpoint/breakpoint in multiple threads."""
        self.build()
        self.do_thread_actions(
            num_signal_threads=1, num_watchpoint_threads=1, num_breakpoint_threads=1
        )
