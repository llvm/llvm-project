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

        bkpt_id = bkpt.GetID()
        loc_string = f"{bkpt_id}.3"
        self.runCmd(f"break disable {loc_string}")
