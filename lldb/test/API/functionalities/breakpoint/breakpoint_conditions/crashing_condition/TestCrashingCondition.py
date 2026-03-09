"""
Test that we recover gracefully from evaluating a breakpoint condition that crashes.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestCrashingCondition(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_crashing_condition(self):
        """Test that if a condition crashes we stop and recover cleanly"""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.do_crash_condition()

    def do_crash_condition(self):
        """Test that if a condition crashes we stop and recover cleanly"""

        # This function starts a process, "a.out" by default, sets a source
        # breakpoint, runs to it, and returns the thread, process & target.
        # It optionally takes an SBLaunchOption argument if you want to pass
        # arguments or environment variables.
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a start breakpoint here", self.main_source_file
        )

        # Set a breakpoint with a condition that crashes on the next line:
        bad_bkpt = target.BreakpointCreateBySourceRegex(
            "Set the test breakpoint here", self.main_source_file
        )
        self.assertGreater(
            bad_bkpt.GetNumLocations(), 0, "Found locations for our breakpoint"
        )
        bad_bkpt.SetCondition("do_crash()")

        self.runCmd("continue")

        self.assertState(process.state, lldb.eStateStopped)
        self.assertStopReason(thread.stop_reason, lldb.eStopReasonBreakpoint)

        # Now we should be able to continue to the real crash:
        error = lldb.SBError()
        self.runCmd("continue")

        # We should have crashed.
        self.assertState(process.state, lldb.eStateStopped)
        # We don't actually know what stop reason a given system will
        # report - it could be eStopReasonException or eStopReasonSignal
        is_crash = (
            thread.stop_reason == lldb.eStopReasonException
            or thread.stop_reason == lldb.eStopReasonSignal
        )
        self.assertIn(thread.stop_reason, [lldb.eStopReasonException, lldb.eStopReasonSignal], "Ran to the actual crash")
