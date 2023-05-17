"""
Test that apropos env doesn't crash trying to touch the process plugin command
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class AproposWithProcessTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// break here')

    def test_apropos_with_process(self):
        """Test that apropos env doesn't crash trying to touch the process plugin command."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main() after the variables are assigned values.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        self.runCmd('apropos env')
