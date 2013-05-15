# coding=utf8
"""
Test that the expression parser returns proper Unicode strings.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

# this test case fails because of rdar://12991846
# the expression parser does not deal correctly with Unicode expressions
# e.g.
#(lldb) expr L"Hello"
#(const wchar_t [6]) $0 = {
#  [0] = \0\0\0\0
#  [1] = \0\0\0\0
#  [2] = \0\0\0\0
#  [3] = \0\0\0\0
#  [4] = H\0\0\0
#  [5] = e\0\0\0
#}

class Rdar12991846TestCase(TestBase):

    mydir = os.path.join("lang", "cpp", "rdar12991846")

    @unittest2.expectedFailure
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test that the expression parser returns proper Unicode strings."""
        self.buildDsym()
        self.rdar12991846()

    @unittest2.expectedFailure
    @dwarf_test
    def test_with_dwarf(self):
        """Test that the expression parser returns proper Unicode strings."""
        self.buildDwarf()
        self.rdar12991846()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.source = 'main.cpp'
        self.line = line_number(self.source, '// Set break point at this line.')

    def rdar12991846(self):
        """Test that the expression parser returns proper Unicode strings."""
        if self.getArchitecture() in ['i386']:
            self.skipTest("Skipping because this test is known to crash on i386")

        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Break on the struct declration statement in main.cpp.
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        if not process:
            self.fail("SBTarget.Launch() failed")

        self.expect('expression L"hello"',
            substrs = ['hello'])

        self.expect('expression u"hello"',
           substrs = ['hello'])

        self.expect('expression U"hello"',
            substrs = ['hello'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
