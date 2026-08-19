"""Simulate MSVC STL std::valarray and check the formatter."""

from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlValarraySimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        va = self.frame().FindVariable("va")
        self.assertEqual(va.GetNumChildren(), 4)
        self.assertEqual(va.GetChildAtIndex(0).GetValueAsSigned(), 1)
        self.assertEqual(va.GetChildAtIndex(3).GetValueAsSigned(), 1234)
        self.expect("frame variable va", substrs=["size=4"])

        va_empty = self.frame().FindVariable("va_empty")
        self.assertEqual(va_empty.GetNumChildren(), 0)
        self.expect("frame variable va_empty", substrs=["size=0"])

        va_ref = self.frame().FindVariable("va_ref")
        self.assertEqual(va_ref.GetNumChildren(), 4)
        self.assertEqual(va_ref.GetChildAtIndex(3).GetValueAsSigned(), 1234)
        self.expect("frame variable va_ref", substrs=["size=4"])
