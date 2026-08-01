"""
Test lldb-dap breakpointLocations request
"""

import os

from lldbsuite.test.decorators import (
    skipIfTargetDoesNotSupportSharedLibraries,
    skipIfWindows,
)
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import BreakpointLocation, LaunchArgs


@skipIfTargetDoesNotSupportSharedLibraries()
class TestDAP_breakpointLocations(DAPTestCaseBase):
    @skipIfWindows
    def test_column_breakpoints(self):
        """Test retrieving the available breakpoint locations."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        main_path = os.path.realpath(self.getBuildArtifact("main-copy.cpp"))

        process_event = session.launch(LaunchArgs(program, stopOnEntry=True))
        session.verify_stopped_on_entry(after=process_event)

        # Ask for the breakpoint locations based only on the line number.
        loop_line = line_number(main_path, "// break loop")
        response = session.get_breakpoint_locations(main_path, loop_line)
        breakpoint_locations = response.body.breakpoints

        expected_locations = [
            BreakpointLocation(line=loop_line, column=9),
            BreakpointLocation(line=loop_line, column=13),
            BreakpointLocation(line=loop_line, column=20),
            BreakpointLocation(line=loop_line, column=23),
            BreakpointLocation(line=loop_line, column=25),
            BreakpointLocation(line=loop_line, column=34),
            BreakpointLocation(line=loop_line, column=37),
            BreakpointLocation(line=loop_line, column=39),
            BreakpointLocation(line=loop_line, column=51),
        ]
        self.assertEqual(breakpoint_locations, expected_locations)

        # Ask for the breakpoint locations for a column range.
        response = session.get_breakpoint_locations(
            main_path, loop_line, column=24, endColumn=46
        )
        breakpoint_locations = response.body.breakpoints
        expected_locations = [
            BreakpointLocation(line=loop_line, column=25),
            BreakpointLocation(line=loop_line, column=34),
            BreakpointLocation(line=loop_line, column=37),
            BreakpointLocation(line=loop_line, column=39),
        ]
        self.assertEqual(breakpoint_locations, expected_locations)

        # Ask for the breakpoint locations for a range of line numbers.
        response = session.get_breakpoint_locations(
            main_path, line=loop_line, column=39, endLine=loop_line + 2
        )
        self.maxDiff = None
        # On some systems, there is an additional breakpoint available
        # at loop_line + 1, column 3, i.e. at the end of the loop. To make
        # this test more portable, only check that all expected breakpoints
        # are presented, but also accept additional breakpoints.
        expected_locations = [
            BreakpointLocation(line=loop_line, column=39),
            BreakpointLocation(line=loop_line, column=51),
            BreakpointLocation(line=loop_line + 2, column=3),
            BreakpointLocation(line=loop_line + 2, column=18),
        ]
        breakpoint_locations = response.body.breakpoints
        for bp in expected_locations:
            self.assertIn(bp, breakpoint_locations)

        session.continue_to_exit()
