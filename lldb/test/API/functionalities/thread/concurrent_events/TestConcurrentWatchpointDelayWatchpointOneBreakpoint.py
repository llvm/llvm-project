from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentWatchpointDelayWatchpointOneBreakpoint(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test two threads that trigger a watchpoint (one with a 1 second delay) and one breakpoint thread."""
        self.build()
        self.do_thread_actions(
            num_watchpoint_threads=1,
            num_delay_watchpoint_threads=1,
            num_breakpoint_threads=1,
        )
