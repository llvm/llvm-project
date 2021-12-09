"""
Test the 'memory read' command.
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
    def test_memory_read_c_string(self):
        """Test that reading memory as a c string respects the size limit given
           and warns if the null terminator is missing."""
        self.build_run_stop()

        # The size here is the size in memory so it includes the null terminator.
        cmd = "memory read --format \"c-string\" --size {} &the_string"

        # Size matches the size of the array.
        self.expect(cmd.format(5), substrs=['\"abcd\"'])

        # If size would take us past the terminator we stop at the terminator.
        self.expect(cmd.format(10), substrs=['\"abcd\"'])

        # Size 3 means 2 chars and a terminator. So we print 2 chars but warn because
        # the third isn't 0 as expected.
        self.expect(cmd.format(3), substrs=['\"ab\"'])
        self.assertRegex(self.res.GetError(),
            "unable to find a NULL terminated string at 0x[0-9A-fa-f]+."
            " Consider increasing the maximum read length.")
