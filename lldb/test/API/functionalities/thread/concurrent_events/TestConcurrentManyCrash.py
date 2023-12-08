from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentManyCrash(ConcurrentEventsBase):
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple="^mips")
    @skipIfOutOfTreeDebugserver
    def test(self):
        """Test 100 threads that cause a segfault."""
        self.build()
        self.do_thread_actions(num_crash_threads=100)
