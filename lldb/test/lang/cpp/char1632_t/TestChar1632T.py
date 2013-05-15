# coding=utf8
"""
Test that the C++11 support for char16_t and char32_t works correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class Char1632TestCase(TestBase):

    mydir = os.path.join("lang", "cpp", "char1632_t")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test that the C++11 support for char16_t and char32_t works correctly."""
        self.buildDsym()
        self.char1632()

    @dwarf_test
    def test_with_dwarf(self):
        """Test that the C++11 support for char16_t and char32_t works correctly."""
        self.buildDwarf()
        self.char1632()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.source = 'main.cpp'
        self.line = line_number(self.source, '// Set break point at this line.')

    def char1632(self):
        """Test that the C++11 support for char16_t and char32_t works correctly."""
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

        if self.TraceOn():
             self.runCmd("frame variable")

        # Check that we correctly report the const types
        self.expect("frame variable cs16 cs32",
            substrs = ['(const char16_t *) cs16 = ','(const char32_t *) cs32 = ','u"hello world ྒྙྐ"','U"hello world ྒྙྐ"'])

        # Check that we correctly report the non-const types
        self.expect("frame variable s16 s32",
            substrs = ['(char16_t *) s16 = ','(char32_t *) s32 = ','u"ﺸﺵۻ"','U"ЕЙРГЖО"'])

        self.runCmd("next") # step to after the string is nullified

        # check that we don't crash on NULL
        self.expect("frame variable s32",
            substrs = ['(char32_t *) s32 = 0x00000000'])

        self.runCmd("next")
        self.runCmd("next")

        # check that the new strings show
        self.expect("frame variable s16 s32",
            substrs = ['(char16_t *) s16 = 0x','(char32_t *) s32 = ','"色ハ匂ヘト散リヌルヲ"','"෴"'])

        # Check that we can run expressions that return charN_t
        self.expect("expression u'a'",substrs = ['(char16_t) $',"61 u'a'"])
        self.expect("expression U'a'",substrs = ['(char32_t) $',"61 U'a'"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
