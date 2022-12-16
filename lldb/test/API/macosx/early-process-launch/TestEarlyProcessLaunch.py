"""Test that we don't read objc class tables early in process startup."""


import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestEarlyProcessLaunch(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIfAsan # rdar://103359354
    @skipIfOutOfTreeDebugserver  # 2022-12-13 FIXME: skipping system debugserver 
                                 # until this feature is included in the system
                                 # debugserver.
    @add_test_categories(['pyapi'])
    def test_early_process_launch(self):
        """Test that we don't read objc class tables early in proc startup"""
        self.build()

        ###
        ### Hit a breakpoint on the first malloc() call, which 
        ### is before libSystem has finished initializing.  At
        ### this point, we should not read the objc class tables.
        ### Then continue to main(), which is past libSystem 
        ### initializing.  Try again, and they should be read.
        ### 
        ### Use the types logging to detect the difference.

        target, process, _, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'malloc')

        target.DisableAllBreakpoints()
        target.BreakpointCreateByName("main")

        logfile_early = os.path.join(self.getBuildDir(), "types-log-early.txt")
        self.addTearDownHook(lambda: self.runCmd("log disable lldb types"))
        self.runCmd("log enable -f %s lldb types" % logfile_early)
        self.runCmd("p global = 15")

        err = process.Continue()
        self.assertTrue(err.Success())

        logfile_later = os.path.join(self.getBuildDir(), "types-log-later.txt")
        self.runCmd("log enable -f %s lldb types" % logfile_later)
        self.runCmd("p global = 25")

        self.assertTrue(os.path.exists(logfile_early))
        self.assertTrue(os.path.exists(logfile_later))
        early_text = open(logfile_early).read()
        later_text = open(logfile_later).read()

        self.assertIn("ran: no, retry: yes", early_text)
        self.assertNotIn("ran: no, retry: yes", later_text)

        self.assertNotIn("ran: yes, retry: no", early_text)
        self.assertIn("ran: yes, retry: no", later_text)
