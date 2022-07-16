
import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentBreakpointsDelayedBreakpointOneWatchpoint(
        ConcurrentEventsBase):

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test a breakpoint, a delayed breakpoint, and one watchpoint thread. """
        self.build()
        self.do_thread_actions(num_breakpoint_threads=1,
                               num_delay_breakpoint_threads=1,
                               num_watchpoint_threads=1)
