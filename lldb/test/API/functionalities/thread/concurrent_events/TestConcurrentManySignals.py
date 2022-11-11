
from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentManySignals(ConcurrentEventsBase):

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    # This test is flaky on Darwin.
    @skipIfDarwin
    @expectedFailureNetBSD
    @skipIfOutOfTreeDebugserver
    def test(self):
        """Test 100 signals from 100 threads."""
        self.build()
        self.do_thread_actions(num_signal_threads=100)
