
import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentBreakpointDelayBreakpointOneSignal(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    def test(self):
        """Test two threads that trigger a breakpoint (one with a 1 second delay) and one signal thread. """
        self.build()
        self.do_thread_actions(num_breakpoint_threads=1,
                               num_delay_breakpoint_threads=1,
                               num_signal_threads=1)
