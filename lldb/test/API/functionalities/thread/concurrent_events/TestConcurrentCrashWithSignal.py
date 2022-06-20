
import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentCrashWithSignal(ConcurrentEventsBase):

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    def test(self):
        """ Test a thread that crashes while another thread generates a signal."""
        self.build()
        self.do_thread_actions(num_crash_threads=1, num_signal_threads=1)
