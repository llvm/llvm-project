"""
Tests that bool types work
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CPPBoolTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_run_command(self):
        """Test that bool types work in the expression parser"""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        line = line_number('main.cpp', '// breakpoint 1')
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", line, num_expected_locations=-1, loc_exact=False)

        self.runCmd("process launch", RUN_SUCCEEDED)

        self.expect_expr("bool second_bool = my_bool; second_bool", result_type="bool", result_value="false")
        self.expect_expr("my_bool = true", result_type="bool", result_value="true")
