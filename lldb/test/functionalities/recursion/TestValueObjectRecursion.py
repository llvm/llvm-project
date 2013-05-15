"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ValueObjectRecursionTestCase(TestBase):

    mydir = os.path.join("functionalities", "recursion")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test that deeply nested ValueObjects still work."""
        self.buildDsym()
        self.recursive_vo_commands()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test that deeply nested ValueObjects still work."""
        self.buildDwarf()
        self.recursive_vo_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def recursive_vo_commands(self):
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
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        root = self.frame().FindVariable("root")
        child = root.GetChildAtIndex(1)
        if self.TraceOn():
             print root
             print child
        for i in range(0,24500):
             child = child.GetChildAtIndex(1)
        if self.TraceOn():
             print child
        self.assertTrue(child.IsValid(),"could not retrieve the deep ValueObject")
        self.assertTrue(child.GetChildAtIndex(0).IsValid(),"the deep ValueObject has no value")
        self.assertTrue(child.GetChildAtIndex(0).GetValueAsUnsigned() != 0,"the deep ValueObject has a zero value")
        self.assertTrue(child.GetChildAtIndex(1).GetValueAsUnsigned() != 0, "the deep ValueObject has no next")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
