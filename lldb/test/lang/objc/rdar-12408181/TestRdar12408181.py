"""
Test that we are able to find out how many children NSWindow has
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class Rdar12408181TestCase(TestBase):

    mydir = os.path.join("lang", "objc", "rdar-12408181")

    @dsym_test
    def test_nswindow_count_with_dsym(self):
        """Test that we are able to find out how many children NSWindow has."""
        d = {'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.nswindow_count(self.exe_name)

    @dwarf_test
    def test_nswindow_count_with_dwarf(self):
        """Test that we are able to find out how many children NSWindow has."""
        d = {'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.nswindow_count(self.exe_name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break inside main().
        self.main_source = "main.m"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    def nswindow_count(self, exe_name):
        """Test that we are able to find out how many children NSWindow has."""
        exe = os.path.join(os.getcwd(), exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, self.main_source, self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        window = self.frame().FindVariable("window")
        window_dynamic = window.GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertTrue(window.GetNumChildren() > 1, "NSWindow (static) only has 1 child!")
        self.assertTrue(window_dynamic.GetNumChildren() > 1, "NSWindow (dynamic) only has 1 child!")
        self.assertTrue(window.GetChildAtIndex(0).IsValid(), "NSWindow (static) has an invalid child")
        self.assertTrue(window_dynamic.GetChildAtIndex(0).IsValid(), "NSWindow (dynamic) has an invalid child")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
