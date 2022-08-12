
from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentTwoWatchpointsOneSignal(ConcurrentEventsBase):

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @expectedFailureNetBSD
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test two threads that trigger a watchpoint and one signal thread. """
        self.build()
        self.do_thread_actions(num_watchpoint_threads=2, num_signal_threads=1)
