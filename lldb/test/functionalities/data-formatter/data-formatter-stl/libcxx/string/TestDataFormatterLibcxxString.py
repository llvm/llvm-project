#coding=utf8
"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class LibcxxStringDataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-stl", "libcxx", "string")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    @skipIfLinux # No standard locations for libc++ on Linux, so skip for now 
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

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=-1)

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
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd("settings set target.max-children-count 256", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect("frame variable",
                    substrs = ['(std::__1::wstring) s = L"hello world! מזל טוב!"',
                    '(std::__1::wstring) S = L"!!!!"',
                    '(const wchar_t *) mazeltov = 0x','L"מזל טוב"',
                    '(std::__1::string) q = "hello world"',
                    '(std::__1::string) Q = "quite a long std::strin with lots of info inside it"'])

        self.runCmd("n")

        self.expect("frame variable",
                    substrs = ['(std::__1::wstring) s = L"hello world! מזל טוב!"',
                    '(std::__1::wstring) S = L"!!!!!"',
                    '(const wchar_t *) mazeltov = 0x','L"מזל טוב"',
                    '(std::__1::string) q = "hello world"',
                    '(std::__1::string) Q = "quite a long std::strin with lots of info inside it"'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
