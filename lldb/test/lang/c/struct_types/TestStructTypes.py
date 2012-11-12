"""
Test that break on a struct declaration has no effect.

Instead, the first executable statement is set as the breakpoint.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class StructTypesTestCase(TestBase):

    mydir = os.path.join("lang", "c", "struct_types")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test that break on a struct declaration has no effect."""
        self.buildDsym()
        self.struct_types()

    @dwarf_test
    def test_with_dwarf(self):
        """Test that break on a struct declaration has no effect."""
        self.buildDwarf()
        self.struct_types()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.c', '// Set break point at this line.')
        self.first_executable_line = line_number('main.c',
                                                 '// This is the first executable statement.')

    def struct_types(self):
        """Test that break on a struct declaration has no effect."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on the struct declration statement in main.c.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped on the first executable statement within the
        # function where the original breakpoint was attempted.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['main.c:%d' % self.first_executable_line,
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # The padding should be an array of size 0
        self.expect("image lookup -t point_tag",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ['padding[0]'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
