"""Simulate MSVC STL std::source_location and check the formatter."""

from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlSourceLocationSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "frame variable loc",
            substrs=['"main.cpp":6:1', "__cdecl", "main"],
        )
        loc_empty = self.frame().FindVariable("loc_empty")
        self.assertTrue(loc_empty.GetError().Success())
        self.assertTrue(not loc_empty.summary)
