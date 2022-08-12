
from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentDelayWatchBreak(ConcurrentEventsBase):

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test (1-second delay) watchpoint and a breakpoint in multiple threads."""
        self.build()
        self.do_thread_actions(
            num_breakpoint_threads=1,
            num_delay_watchpoint_threads=1)
