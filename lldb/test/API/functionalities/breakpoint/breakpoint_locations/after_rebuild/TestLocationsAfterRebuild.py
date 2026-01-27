"""
When a rebuild causes a location to be removed, make sure
we still handle the remaining locations correctly.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import skipIfWindows
import os


class TestLocationsAfterRebuild(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    # On Windows we cannot remove a file that lldb is debugging.
    @skipIfWindows
    def test_remaining_location_spec(self):
        """If we rebuild a couple of times some of the old locations
        get removed.  Make sure the command-line breakpoint id
        validator still works correctly."""
        self.build(dictionary={"C_SOURCES": "main.c", "EXE": "a.out"})

        path_to_exe = self.getBuildArtifact()

        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(self, "main")

        # Let the process continue to exit:
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited, "Ran to completion")
        os.remove(path_to_exe)

        # We have to rebuild twice with changed sources to get
        # us to remove the first set of locations:
        self.build(dictionary={"C_SOURCES": "second_main.c", "EXE": "a.out"})

        (target, process, thread, bkpt) = lldbutil.run_to_breakpoint_do_run(
            self, target, bkpt
        )

        # Let the process continue to exit:
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited, "Ran to completion")

        os.remove(path_to_exe)

        self.build(dictionary={"C_SOURCES": "third_main.c", "EXE": "a.out"})

        (target, process, thread, bkpt) = lldbutil.run_to_breakpoint_do_run(
            self, target, bkpt
        )

        # After enabling locate_module callback for main executables,
        # the number of locations may vary depending on the platform.
        num_locs = bkpt.GetNumLocations()
        bkpt_id = bkpt.GetID()

        self.assertGreater(
            num_locs,
            0,
            f"Expected at least one breakpoint location, but found {num_locs}",
        )

        # Iterate through all valid locations and verify we can disable each one.
        # This tests that breakpoint location IDs remain valid after rebuilds.
        for loc_idx in range(num_locs):
            loc = bkpt.GetLocationAtIndex(loc_idx)
            self.assertTrue(loc.IsValid(), f"Location at index {loc_idx} is not valid")

            # Get the actual location ID from the location object
            loc_id = loc.GetID()
            loc_string = f"{bkpt_id}.{loc_id}"
            self.runCmd(f"break disable {loc_string}")
