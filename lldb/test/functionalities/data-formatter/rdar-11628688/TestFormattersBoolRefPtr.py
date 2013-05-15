"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import datetime
import lldbutil

class DataFormatterOSTypeTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "rdar-11628688")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_ostype_with_dsym_and_run_command(self):
        """Test the formatters we use for OSType."""
        self.buildDsym()
        self.ostype_data_formatter_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_ostype_with_dwarf_and_run_command(self):
        """Test the formatters we use for OSType."""
        self.buildDwarf()
        self.ostype_data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.mm', '// Set break point at this line.')

    def ostype_data_formatter_commands(self):
        """Test the formatters we use for OSType."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.mm", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Now check that we use the right summary for OSType
        self.expect('frame variable',
                    substrs = ["(OSType) a = 1952805748 'test'","(OSType) b = 1650815860 'best'"])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
