"""Simulate MSVC STL std::filesystem::path and check the formatter."""

from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlPathSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect("frame variable p", substrs=["file.txt"])
        empty = self.frame().FindVariable("empty")
        self.assertTrue(empty.GetError().Success())
