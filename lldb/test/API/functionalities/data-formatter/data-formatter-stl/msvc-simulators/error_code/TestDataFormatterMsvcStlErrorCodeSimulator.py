"""Simulate MSVC STL std::error_code and check the formatter."""

from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlErrorCodeSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        ec = self.frame().FindVariable("ec")
        self.assertTrue(ec.IsValid())
        self.assertGreaterEqual(ec.GetNumChildren(), 1)
        self.assertIsNotNone(ec.GetChildMemberWithName("_Mycat"))
        self.expect("frame variable ec", substrs=["value=2"])

        econd = self.frame().FindVariable("econd")
        self.assertTrue(econd.IsValid())
        self.assertGreaterEqual(econd.GetNumChildren(), 1)
        self.expect("frame variable econd", substrs=["value=7"])
