"""
Test lldb-dap breakpointLocations request
"""


import dap_server
import shutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


class TestDAP_breakpointLocations(lldbdap_testcase.DAPTestCaseBase):
    def setUp(self):
        lldbdap_testcase.DAPTestCaseBase.setUp(self)

        self.main_basename = "main-copy.cpp"
        self.main_path = os.path.realpath(self.getBuildArtifact(self.main_basename))

    @skipIfWindows
    def test_column_breakpoints(self):
        """Test retrieving the available breakpoint locations."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        loop_line = line_number(self.main_path, "// break loop")
        self.dap_server.request_continue()

        # Ask for the breakpoint locations based only on the line number
        response = self.dap_server.request_breakpointLocations(
            self.main_path, loop_line
        )
        self.assertTrue(response["success"])
        self.assertEqual(
            response["body"]["breakpoints"],
            [
                {"line": loop_line, "column": 9},
                {"line": loop_line, "column": 13},
                {"line": loop_line, "column": 20},
                {"line": loop_line, "column": 23},
                {"line": loop_line, "column": 25},
                {"line": loop_line, "column": 34},
                {"line": loop_line, "column": 37},
                {"line": loop_line, "column": 39},
                {"line": loop_line, "column": 51},
            ],
        )

        # Ask for the breakpoint locations for a column range
        response = self.dap_server.request_breakpointLocations(
            self.main_path,
            loop_line,
            column=24,
            end_column=46,
        )
        self.assertTrue(response["success"])
        self.assertEqual(
            response["body"]["breakpoints"],
            [
                {"line": loop_line, "column": 25},
                {"line": loop_line, "column": 34},
                {"line": loop_line, "column": 37},
                {"line": loop_line, "column": 39},
            ],
        )

        # Ask for the breakpoint locations for a range of line numbers
        response = self.dap_server.request_breakpointLocations(
            self.main_path,
            line=loop_line,
            end_line=loop_line + 2,
            column=39,
        )
        self.maxDiff = None
        self.assertTrue(response["success"])
        # On some systems, there is an additional breakpoint available
        # at line 41, column 3, i.e. at the end of the loop. To make this
        # test more portable, only check that all expected breakpoints are
        # presented, but also accept additional breakpoints.
        expected_breakpoints = [
            {"column": 39, "line": 40},
            {"column": 51, "line": 40},
            {"column": 3, "line": 42},
            {"column": 18, "line": 42},
        ]
        for bp in expected_breakpoints:
            self.assertIn(bp, response["body"]["breakpoints"])
