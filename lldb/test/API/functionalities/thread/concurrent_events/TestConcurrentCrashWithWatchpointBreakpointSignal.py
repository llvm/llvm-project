
import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentCrashWithWatchpointBreakpointSignal(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @add_test_categories(["watchpoint"])
    def test(self):
        """ Test a thread that crashes while other threads generate a signal and hit a watchpoint and breakpoint. """
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=1,
                               num_breakpoint_threads=1,
                               num_signal_threads=1,
                               num_watchpoint_threads=1)
