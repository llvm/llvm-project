from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentSignalWatch(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test a watchpoint and a signal in multiple threads."""
        self.build()
        self.do_thread_actions(num_signal_threads=1, num_watchpoint_threads=1)
