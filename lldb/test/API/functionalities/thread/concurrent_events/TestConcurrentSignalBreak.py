from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentSignalBreak(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    def test(self):
        """Test signal and a breakpoint in multiple threads."""
        self.build()
        self.do_thread_actions(num_breakpoint_threads=1, num_signal_threads=1)
