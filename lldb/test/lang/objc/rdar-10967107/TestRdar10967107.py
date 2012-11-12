"""
Test that CoreFoundation classes CFGregorianDate and CFRange are not improperly uniqued
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class Rdar10967107TestCase(TestBase):

    mydir = os.path.join("lang", "objc", "rdar-10967107")

    @dsym_test
    def test_cfrange_diff_cfgregoriandate_with_dsym(self):
        """Test that CoreFoundation classes CFGregorianDate and CFRange are not improperly uniqued."""
        d = {'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.cfrange_diff_cfgregoriandate(self.exe_name)

    @dwarf_test
    def test_cfrange_diff_cfgregoriandate_with_dwarf(self):
        """Test that CoreFoundation classes CFGregorianDate and CFRange are not improperly uniqued."""
        d = {'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.cfrange_diff_cfgregoriandate(self.exe_name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break inside main().
        self.main_source = "main.m"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    def cfrange_diff_cfgregoriandate(self, exe_name):
        """Test that CoreFoundation classes CFGregorianDate and CFRange are not improperly uniqued."""
        exe = os.path.join(os.getcwd(), exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, self.main_source, self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        # check that each type is correctly bound to its list of children
        self.expect("frame variable cf_greg_date --raw", substrs = ['year','month','day','hour','minute','second'])
        self.expect("frame variable cf_range --raw", substrs = ['location','length'])
        # check that printing both does not somehow confuse LLDB
        self.expect("frame variable  --raw", substrs = ['year','month','day','hour','minute','second','location','length'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
