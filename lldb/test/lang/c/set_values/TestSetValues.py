"""Test settings and readings of program variables."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class SetValuesTestCase(TestBase):

    mydir = os.path.join("lang", "c", "set_values")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test settings and readings of program variables."""
        self.buildDsym()
        self.set_values()

    @dwarf_test
    def test_with_dwarf(self):
        """Test settings and readings of program variables."""
        self.buildDwarf()
        self.set_values()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.line1 = line_number('main.c', '// Set break point #1.')
        self.line2 = line_number('main.c', '// Set break point #2.')
        self.line3 = line_number('main.c', '// Set break point #3.')
        self.line4 = line_number('main.c', '// Set break point #4.')
        self.line5 = line_number('main.c', '// Set break point #5.')

    def set_values(self):
        """Test settings and readings of program variables."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set breakpoints on several places to set program variables.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line1, num_expected_locations=1, loc_exact=True)

        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line2, num_expected_locations=1, loc_exact=True)

        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line3, num_expected_locations=1, loc_exact=True)

        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line4, num_expected_locations=1, loc_exact=True)

        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line5, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # main.c:15
        # Check that 'frame variable --show-types' displays the correct data type and value.
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(char) i = 'a'")

        # Now set variable 'i' and check that it is correctly displayed.
        self.runCmd("expression i = 'b'")
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(char) i = 'b'")

        self.runCmd("continue")

        # main.c:36
        # Check that 'frame variable --show-types' displays the correct data type and value.
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            patterns = ["\((short unsigned int|unsigned short)\) i = 33"])

        # Now set variable 'i' and check that it is correctly displayed.
        self.runCmd("expression i = 333")
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            patterns = ["\((short unsigned int|unsigned short)\) i = 333"])

        self.runCmd("continue")

        # main.c:57
        # Check that 'frame variable --show-types' displays the correct data type and value.
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(long) i = 33")

        # Now set variable 'i' and check that it is correctly displayed.
        self.runCmd("expression i = 33333")
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(long) i = 33333")

        self.runCmd("continue")

        # main.c:78
        # Check that 'frame variable --show-types' displays the correct data type and value.
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(double) i = 3.14159")

        # Now set variable 'i' and check that it is correctly displayed.
        self.runCmd("expression i = 3.14")
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(double) i = 3.14")

        self.runCmd("continue")

        # main.c:85
        # Check that 'frame variable --show-types' displays the correct data type and value.
        # rdar://problem/8422727
        # set_values test directory: 'frame variable' shows only (long double) i =
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(long double) i = 3.14159")

        # Now set variable 'i' and check that it is correctly displayed.
        self.runCmd("expression i = 3.1")
        self.expect("frame variable --show-types", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(long double) i = 3.1")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
