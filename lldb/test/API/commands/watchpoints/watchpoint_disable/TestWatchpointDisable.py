"""
Test that the SBWatchpoint::SetEnable API works.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbplatform, lldbplatformutil


class TestWatchpointSetEnable(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_disable_works (self):
        """Set a watchpoint, disable it, and make sure it doesn't get hit."""
        self.build()
        self.do_test(False)

    def test_disable_enable_works (self):
        """Set a watchpoint, disable it, and make sure it doesn't get hit."""
        self.build()
        self.do_test(True)

    def do_test(self, test_enable):
        """Set a watchpoint, disable it and make sure it doesn't get hit."""

        main_file_spec = lldb.SBFileSpec("main.c")

        self.target = self.createTestTarget()

        bkpt_before = self.target.BreakpointCreateBySourceRegex("Set a breakpoint here", main_file_spec)
        self.assertEqual(bkpt_before.GetNumLocations(),  1, "Failed setting the before breakpoint.")

        bkpt_after = self.target.BreakpointCreateBySourceRegex("We should have stopped", main_file_spec)
        self.assertEqual(bkpt_after.GetNumLocations(), 1, "Failed setting the after breakpoint.")

        process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = lldbutil.get_one_thread_stopped_at_breakpoint(process, bkpt_before)
        self.assertTrue(thread.IsValid(), "We didn't stop at the before breakpoint.")

        ret_val = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("watchpoint set variable -w write global_var", ret_val)
        self.assertTrue(ret_val.Succeeded(), "Watchpoint set variable did not return success.")

        wp = self.target.FindWatchpointByID(1)
        self.assertTrue(wp.IsValid(), "Didn't make a valid watchpoint.")
        self.assertNotEqual(wp.GetWatchAddress(), lldb.LLDB_INVALID_ADDRESS, "Watch address is invalid")

        wp.SetEnabled(False)
        self.assertTrue(not wp.IsEnabled(), "The watchpoint thinks it is still enabled")

        process.Continue()

        stop_reason = thread.GetStopReason()

        self.assertStopReason(stop_reason, lldb.eStopReasonBreakpoint, "We didn't stop at our breakpoint.")

        if test_enable:
            wp.SetEnabled(True)
            self.assertTrue(wp.IsEnabled(), "The watchpoint thinks it is still disabled.")
            process.Continue()
            stop_reason = thread.GetStopReason()
            self.assertStopReason(stop_reason, lldb.eStopReasonWatchpoint, "We didn't stop at our watchpoint")

