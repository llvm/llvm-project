from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentTwoWatchpointThreads(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test two threads that trigger a watchpoint."""
        self.build()
        self.do_thread_actions(num_watchpoint_threads=2)
