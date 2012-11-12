"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class Radar9973992DataFormatterTestCase(TestBase):

    # test for rdar://problem/9973992 (What should we do for "${var}" in summaries of aggregate types?)
    mydir = os.path.join("functionalities", "data-formatter", "rdar-9973992")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test data formatter commands."""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type summary add --summary-string \"SUMMARY SUCCESS ${var}\" Summarize")
        
        self.expect('frame variable mine_ptr',
                substrs = ['SUMMARY SUCCESS summarize_ptr_t @ '])

        self.expect('frame variable *mine_ptr',
                substrs = ['SUMMARY SUCCESS summarize_t @'])

        self.runCmd("type summary add --summary-string \"SUMMARY SUCCESS ${var.first}\" Summarize")

        self.expect('frame variable mine_ptr',
                    substrs = ['SUMMARY SUCCESS 10'])

        self.expect('frame variable *mine_ptr',
                    substrs = ['SUMMARY SUCCESS 10'])
            
        self.runCmd("type summary add --summary-string \"${var}\" Summarize")
        self.runCmd("type summary add --summary-string \"${var}\" -e TwoSummarizes")
            
        self.expect('frame variable',
            substrs = ['(TwoSummarizes) twos = TwoSummarizes @ ',
                       'first = summarize_t @ ',
                       'second = summarize_t @ '])
                    
        self.runCmd("type summary add --summary-string \"SUMMARY SUCCESS ${var.first}\" Summarize")
        self.expect('frame variable',
                    substrs = ['(TwoSummarizes) twos = TwoSummarizes @ ',
                               'first = SUMMARY SUCCESS 1',
                               'second = SUMMARY SUCCESS 3'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
