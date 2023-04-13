"""
Watch one byte in the middle of a doubleword, mutate the
entire doubleword including the watched byte.  On AArch64
the trap address is probably the start of the doubleword,
instead of the address of our watched byte.  Test that lldb
correctly associates this watchpoint trap with our watchpoint.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class UnalignedWatchpointTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @skipIfOutOfTreeDebugserver
    def test_unaligned_watchpoint(self):
        """Test an unaligned watchpoint triggered by a larger aligned write."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
             "break here", self.main_source_file)

        frame = thread.GetFrameAtIndex(0)

        self.expect("watchpoint set variable a.buf[2]")
        
        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type), but
        # only once.  The stop reason of the thread should be watchpoint.
        self.expect("thread list", STOPPED_DUE_TO_WATCHPOINT,
                    substrs=['stopped',
                             'stop reason = watchpoint'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 1.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 1'])
        
        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type), but
        # only once.  The stop reason of the thread should be watchpoint.
        self.expect("thread list", STOPPED_DUE_TO_WATCHPOINT,
                    substrs=['stopped',
                             'stop reason = watchpoint'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 1.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 2'])
