"""
Test that objective-c method returning BOOL works correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class MethodReturningBOOLTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "radar-9691614")

    @dsym_test
    def test_method_ret_BOOL_with_dsym(self):
        """Test that objective-c method returning BOOL works correctly."""
        d = {'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.objc_method_ret_BOOL(self.exe_name)

    @dwarf_test
    def test_method_ret_BOOL_with_dwarf(self):
        """Test that objective-c method returning BOOL works correctly."""
        d = {'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.objc_method_ret_BOOL(self.exe_name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break inside main().
        self.main_source = "main.m"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    def objc_method_ret_BOOL(self, exe_name):
        """Test that objective-c method returning BOOL works correctly."""
        exe = os.path.join(os.getcwd(), exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
            substrs = [" at %s:%d" % (self.main_source, self.line),
                       "stop reason = breakpoint"])

        # rdar://problem/9691614
        self.runCmd('p (int)[my isValid]')

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
