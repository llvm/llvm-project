from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentDelaySignalWatch(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test a watchpoint and a (1 second delay) signal in multiple threads."""
        self.build()
        self.do_thread_actions(num_delay_signal_threads=1, num_watchpoint_threads=1)
