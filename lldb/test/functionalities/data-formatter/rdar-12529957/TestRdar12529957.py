"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import datetime
import lldbutil

class DataFormatterRdar12529957TestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "rdar-12529957")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_rdar12529957_with_dsym_and_run_command(self):
        """Test that NSSet reports its synthetic children properly."""
        self.buildDsym()
        self.rdar12529957_tester()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_rdar12529957_with_dwarf_and_run_command(self):
        """Test that NSSet reports its synthetic children properly."""
        self.buildDwarf()
        self.rdar12529957_tester()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// Set break point at this line.')

    def rdar12529957_tester(self):
        """Test that NSSet reports its synthetic children properly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

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

        # Now check that we are displaying Cocoa classes correctly
        self.expect('frame variable set',
                    substrs = ['4 objects'])
        self.expect('frame variable mutable',
                    substrs = ['9 objects'])
        self.expect('frame variable set --ptr-depth 1 -d run -T',
                    substrs = ['4 objects','[0]','[1]','[2]','[3]','hello','world','(int)1','(int)2'])
        self.expect('frame variable mutable --ptr-depth 1 -d run -T',
                    substrs = ['9 objects','(int)5','@"3 objects"','@"www.apple.com"','(int)3','@"world"','(int)4'])

        self.runCmd("next")
        self.expect('frame variable mutable',
                    substrs = ['0 objects'])

        self.runCmd("next")
        self.expect('frame variable mutable',
                    substrs = ['4 objects'])
        self.expect('frame variable mutable --ptr-depth 1 -d run -T',
                    substrs = ['4 objects','[0]','[1]','[2]','[3]','hello','world','(int)1','(int)2'])

        self.runCmd("next")
        self.expect('frame variable mutable',
                    substrs = ['4 objects'])
        self.expect('frame variable mutable --ptr-depth 1 -d run -T',
                    substrs = ['4 objects','[0]','[1]','[2]','[3]','hello','world','(int)1','(int)2'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
