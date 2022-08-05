import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentManyWatchpoints(ConcurrentEventsBase):

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @add_test_categories(["watchpoint"])
    @skipIfOutOfTreeDebugserver
    def test(self):
        """Test 100 watchpoints from 100 threads."""
        self.build()
        self.do_thread_actions(num_watchpoint_threads=100)
