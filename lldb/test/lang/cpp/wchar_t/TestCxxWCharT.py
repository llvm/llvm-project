#coding=utf8
"""
Test that C++ supports wchar_t correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class CxxWCharTTestCase(TestBase):

    mydir = os.path.join("lang", "cpp", "wchar_t")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test that C++ supports wchar_t correctly."""
        self.buildDsym()
        self.wchar_t()

    @dwarf_test
    def test_with_dwarf(self):
        """Test that C++ supports wchar_t correctly."""
        self.buildDwarf()
        self.wchar_t()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.source = 'main.cpp'
        self.line = line_number(self.source, '// Set break point at this line.')

    def wchar_t(self):
        """Test that C++ supports wchar_t correctly."""
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

        # Check that we correctly report templates on wchar_t
        self.expect("frame variable foo_y",
            substrs = ['(Foo<wchar_t>) foo_y = '])

        # Check that we correctly report templates on int
        self.expect("frame variable foo_x",
            substrs = ['(Foo<int>) foo_x = '])

        # Check that we correctly report wchar_t
        self.expect("frame variable foo_y.object",
            substrs = ['(wchar_t) foo_y.object = '])

        # Check that we correctly report int
        self.expect("frame variable foo_x.object",
            substrs = ['(int) foo_x.object = '])

        # Check that we can run expressions that return wchar_t
        self.expect("expression L'a'",substrs = ['(wchar_t) $',"61 L'a'"])

        # Mazel Tov if this works!
        self.expect("frame variable mazeltov",
            substrs = ['(const wchar_t *) mazeltov = ','L"מזל טוב"'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
