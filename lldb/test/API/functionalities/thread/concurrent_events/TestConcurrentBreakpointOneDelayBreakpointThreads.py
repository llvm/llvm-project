from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentBreakpointOneDelayBreakpointThreads(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    def test(self):
        """Test threads that trigger a breakpoint where one thread has a 1 second delay."""
        self.build()
        self.do_thread_actions(num_breakpoint_threads=1, num_delay_breakpoint_threads=1)
