"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import datetime
import lldbutil

class DataFormatterRdar11988289TestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "rdar-11988289")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_rdar11988289_with_dsym_and_run_command(self):
        """Test that NSDictionary reports its synthetic children properly."""
        self.buildDsym()
        self.rdar11988289_tester()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_rdar11988289_with_dwarf_and_run_command(self):
        """Test that NSDictionary reports its synthetic children properly."""
        self.buildDwarf()
        self.rdar11988289_tester()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// Set break point at this line.')

    def rdar11988289_tester(self):
        """Test that NSDictionary reports its synthetic children properly."""
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
        self.expect('frame variable dictionary',
                    substrs = ['3 key/value pairs'])
        self.expect('frame variable mutable',
                    substrs = ['4 key/value pairs'])
        self.expect('frame variable dictionary --ptr-depth 1',
                    substrs = ['3 key/value pairs','[0] = {','key = 0x','value = 0x','[1] = {','[2] = {'])
        self.expect('frame variable mutable --ptr-depth 1',
                    substrs = ['4 key/value pairs','[0] = {','key = 0x','value = 0x','[1] = {','[2] = {','[3] = {'])
        self.expect('frame variable dictionary --ptr-depth 1 -d no-run-target',
                    substrs = ['3 key/value pairs','@"bar"','@"2 objects"','@"baz"','2 key/value pairs'])
        self.expect('frame variable mutable --ptr-depth 1 -d no-run-target',
                    substrs = ['4 key/value pairs','(int)23','@"123"','@"http://www.apple.com"','@"puartist"','3 key/value pairs'])
        self.expect('frame variable mutable --ptr-depth 2 -d no-run-target',
        substrs = ['4 key/value pairs','(int)23','@"123"','@"http://www.apple.com"','@"puartist"','3 key/value pairs {','@"bar"','@"2 objects"'])
        self.expect('frame variable mutable --ptr-depth 3 -d no-run-target',
        substrs = ['4 key/value pairs','(int)23','@"123"','@"http://www.apple.com"','@"puartist"','3 key/value pairs {','@"bar"','@"2 objects"','(int)1','@"two"'])

        self.assertTrue(self.frame().FindVariable("dictionary").MightHaveChildren(), "dictionary says it does not have children!")
        self.assertTrue(self.frame().FindVariable("mutable").MightHaveChildren(), "mutable says it does not have children!")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
