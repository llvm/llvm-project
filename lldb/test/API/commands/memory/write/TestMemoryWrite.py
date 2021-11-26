"""
Test the 'memory write' command.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class MemoryWriteTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def build_run_stop(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main() after the variables are assigned values.
        lldbutil.run_break_set_by_file_and_line(self,
                                                "main.c",
                                                self.line,
                                                num_expected_locations=1,
                                                loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list",
                    STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

    @no_debug_info_test
    def test_memory_write(self):
        """Test the 'memory write' command for writing values and file contents."""
        self.build_run_stop()

        self.expect(
            "memory read --format c --size 7 --count 1 `&my_string`",
            substrs=['abcdefg'])

        self.expect(
            "memory write --format c --size 7 `&my_string` ABCDEFG")

        self.expect(
            "memory read --format c --size 7 --count 1 `&my_string`",
            substrs=['ABCDEFG'])

        self.expect(
            "memory write --infile file.txt --size 7 `&my_string`",
            substrs=['7 bytes were written'])

        self.expect(
            "memory read --format c --size 7 --count 1 `&my_string`",
            substrs=['abcdefg'])

        self.expect(
            "memory write --infile file.txt --size 7 `&my_string` ABCDEFG", error=True,
            substrs=['error: memory write takes only a destination address when writing file contents'])

        self.expect(
            "memory write --infile file.txt --size 7", error=True,
            substrs=['error: memory write takes a destination address when writing file contents'])

    @no_debug_info_test
    def test_memory_write_command_usage_syntax(self):
        """Test that 'memory write' command usage syntax shows it does not take values when writing file contents."""
        self.expect(
            "help memory write",
            substrs=[
                "memory write [-f <format>] [-s <byte-size>] <address> <value> [<value> [...]]",
                "memory write -i <filename> [-s <byte-size>] [-o <offset>] <address>"])
