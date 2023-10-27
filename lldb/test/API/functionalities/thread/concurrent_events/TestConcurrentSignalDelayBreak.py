from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentSignalDelayBreak(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @expectedFlakeyNetBSD
    def test(self):
        """Test signal and a (1 second delay) breakpoint in multiple threads."""
        self.build()
        self.do_thread_actions(num_delay_breakpoint_threads=1, num_signal_threads=1)
