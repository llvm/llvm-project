
from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentCrashWithBreak(ConcurrentEventsBase):

    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    def test(self):
        """ Test a thread that crashes while another thread hits a breakpoint."""
        self.build()
        self.do_thread_actions(num_crash_threads=1, num_breakpoint_threads=1)
