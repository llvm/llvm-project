"""
debugserver used to block replying to the 'D' packet
till it had joined the profiling thread.  If the profiling interval
was too long, that would mean it would take longer than the packet
timeout to reply, and the detach would time out.  Make sure that doesn't
happen.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import os
import signal

class TestDetachVrsProfile(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIfOutOfTreeDebugserver
    @skipIfRemote
    def test_profile_and_detach(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.do_profile_and_detach()

    def do_profile_and_detach(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        # First make sure we are getting async data.  Set a short interval, continue a bit and check:
        interp.HandleCommand("process plugin packet send 'QSetEnableAsyncProfiling;enable:1;interval_usec:500000;scan_type=0x5;'", result)
        self.assertTrue(result.Succeeded(), "process packet send failed: %s"%(result.GetError()))

        # Run a bit to give us a change to collect profile data:
        bkpt.SetIgnoreCount(1)
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Hit our breakpoint again.")
        str = process.GetAsyncProfileData(1000)
        self.assertTrue(len(str) > 0, "Got some profile data")

        # Now make the profiling interval very long and try to detach.
        interp.HandleCommand("process plugin packet send 'QSetEnableAsyncProfiling;enable:1;interval_usec:10000000;scan_type=0x5;'", result)
        self.assertTrue(result.Succeeded(), "process packet send failed: %s"%(result.GetError()))
        self.dbg.SetAsync(True)
        listener = self.dbg.GetListener()

        # We don't want to hit our breakpoint anymore.
        bkpt.SetEnabled(False)

        # Record our process pid so we can kill it since we are going to detach...
        self.pid = process.GetProcessID()
        def cleanup():
            self.dbg.SetAsync(False)
            os.kill(self.pid, signal.SIGKILL)
        self.addTearDownHook(cleanup)

        process.Continue()

        event = lldb.SBEvent()
        success = listener.WaitForEventForBroadcaster(0, process.GetBroadcaster(), event)
        self.assertTrue(success, "Got an event which should be running.")
        event_state = process.GetStateFromEvent(event)
        self.assertEqual(event_state, lldb.eStateRunning, "Got the running event")

        # Now detach:
        error = process.Detach()
        self.assertSuccess(error, "Detached successfully")
