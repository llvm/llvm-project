"""Simulate MSVC STL std::expected and check the formatter."""

from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlExpectedSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        ok = self.frame().FindVariable("ok")
        self.assertEqual(ok.GetNumChildren(), 1)
        self.assertEqual(ok.GetChildAtIndex(0).GetName(), "Value")
        self.assertEqual(ok.GetChildAtIndex(0).GetValueAsSigned(), 7)
        self.assertFalse(ok.GetChildMemberWithName("Unexpected").IsValid())
        self.expect("frame variable ok", substrs=["Has Value=true", "Value = 7"])

        err = self.frame().FindVariable("err")
        self.assertEqual(err.GetNumChildren(), 1)
        self.assertEqual(err.GetChildAtIndex(0).GetName(), "Unexpected")
        self.assertFalse(err.GetChildMemberWithName("Value").IsValid())
        self.expect("frame variable err", substrs=["Has Value=false", "boom"])

        void_ok = self.frame().FindVariable("void_ok")
        self.assertEqual(void_ok.GetNumChildren(), 0)
        self.expect("frame variable void_ok", substrs=["Has Value=true"])

        void_err = self.frame().FindVariable("void_err")
        self.assertEqual(void_err.GetNumChildren(), 1)
        self.assertEqual(void_err.GetChildAtIndex(0).GetValueAsSigned(), 11)
        self.expect("frame variable void_err", substrs=["Has Value=false"])
