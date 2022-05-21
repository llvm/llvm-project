"""
Test that by turning off EXC_BAD_ACCESS catching, we can
debug into and out of a signal handler.
"""

import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class TestDarwinSignalHandlers(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_ignored_thread(self):
        """It isn't possible to convert an EXC_BAD_ACCESS to a signal when
        running under the debugger, which makes debugging SIGBUS handlers
        and so forth difficult.  This test sends QIgnoreExceptions and that
        should get us into the signal handler and out again. """
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.suspended_thread_test()

    def suspended_thread_test(self):
        # Make sure that we don't accept bad values:
        self.match("settings set platform.plugin.darwin.ignored-exceptions EXC_BAD_AXESS", "EXC_BAD_AXESS", error=True)
        # Now set ourselves to ignore some exceptions.  The test depends on ignoring EXC_BAD_ACCESS, but I passed a couple
        # to make sure they parse:
        self.runCmd("settings set platform.plugin.darwin.ignored-exceptions EXC_BAD_ACCESS|EXC_ARITHMETIC")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Stop here to get things going", self.main_source_file)

        sig_bkpt = target.BreakpointCreateBySourceRegex("stop here in the signal handler",
                                                        self.main_source_file)
        self.assertEqual(sig_bkpt.GetNumLocations(), 1, "Found sig handler breakpoint")
        return_bkpt = target.BreakpointCreateBySourceRegex("Break here to make sure we got past the signal handler",
                                                        self.main_source_file)
        self.assertEqual(return_bkpt.GetNumLocations(), 1, "Found return breakpoint")
        # Now continue, and we should stop with a stop reason of SIGBUS:
        process.Continue()
        self.assertEqual(process.state, lldb.eStateStopped, "Stopped after continue to SIGBUS")
        if thread.stop_reason == lldb.eStopReasonBreakpoint:
            id = thread.GetStopReasonDataAtIndex(0)
            name = thread.frame[0].name
            self.fail("Hit breakpoint {0} in '{1}' rather than getting a SIGBUS".format(id, name))

        self.assertEqual(thread.stop_reason, lldb.eStopReasonSignal)
        self.assertEqual(thread.GetStopReasonDataAtIndex(0), 10, "Got a SIGBUS")

        # Now when we continue, we'll find our way into the signal handler:
        threads = lldbutil.continue_to_breakpoint(process, sig_bkpt)
        self.assertEqual(len(threads), 1, "Stopped at sig breakpoint")

        threads = lldbutil.continue_to_breakpoint(process, return_bkpt)
        self.assertEqual(len(threads), 1, "Stopped at return breakpoint")

        # Make sure we really changed the value:
        
        process.Continue()
        self.assertEqual(process.state, lldb.eStateExited, "Process exited")
        self.assertEqual(process.exit_state, 20, "Got the right exit status")
                         
