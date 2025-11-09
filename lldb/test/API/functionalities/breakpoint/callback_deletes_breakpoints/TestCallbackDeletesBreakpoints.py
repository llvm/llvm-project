"""
Make sure that deleting breakpoints in another breakpoint
callback doesn't cause problems.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestBreakpointDeletionInCallback(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_breakpoint_deletion_in_callback(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.delete_others_test()

    def delete_others_test(self):
        """You might use the test implementation in several ways, say so here."""

        # This function starts a process, "a.out" by default, sets a source
        # breakpoint, runs to it, and returns the thread, process & target.
        # It optionally takes an SBLaunchOption argument if you want to pass
        # arguments or environment variables.
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        # Now set a breakpoint on "I did something" several times
        #
        bkpt_numbers = []
        for idx in range(0, 5):
            bkpt_numbers.append(
                lldbutil.run_break_set_by_source_regexp(self, "// Deletable location")
            )

        # And add commands to the third one to delete two others:
        deleter = target.FindBreakpointByID(bkpt_numbers[2])
        self.assertTrue(deleter.IsValid(), "Deleter is a good breakpoint")
        commands = lldb.SBStringList()
        deleted_ids = [bkpt_numbers[0], bkpt_numbers[3]]
        for idx in deleted_ids:
            commands.AppendString(f"break delete {idx}")

        deleter.SetCommandLineCommands(commands)

        thread_list = lldbutil.continue_to_breakpoint(process, deleter)
        self.assertEqual(len(thread_list), 1)
        stop_data = thread.stop_reason_data
        # There are 5 breakpoints so 10 break_id, break_loc_id.
        self.assertEqual(len(stop_data), 10)
        # We should have been able to get break ID's and locations for all the
        # breakpoints that we originally hit, but some won't be around anymore:
        for idx in range(0, 5):
            bkpt_id = stop_data[idx * 2]
            print(f"{idx}: {bkpt_id}")
            self.assertIn(bkpt_id, bkpt_numbers, "Found breakpoints are right")
            loc_id = stop_data[idx * 2 + 1]
            self.assertEqual(loc_id, 1, "All breakpoints have one location")
            bkpt = target.FindBreakpointByID(bkpt_id)
            if bkpt_id in deleted_ids:
                # Looking these up should be an error:
                self.assertFalse(bkpt.IsValid(), "Deleted breakpoints are deleted")
            else:
                self.assertTrue(bkpt.IsValid(), "The rest are still valid")
