
import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentWatchBreak(ConcurrentEventsBase):

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @add_test_categories(["watchpoint"])
    @skipIf(
    oslist=["ios", "watchos", "tvos", "bridgeos", "macosx"],
    archs=['arm64', 'arm64e', 'arm64_32', 'arm'],
    bugnumber="rdar://93863107")

    def test(self):
        """Test watchpoint and a breakpoint in multiple threads."""
        self.build()
        self.do_thread_actions(
            num_breakpoint_threads=1,
            num_watchpoint_threads=1)
