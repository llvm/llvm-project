"""Simulate MSVC STL layouts and check the corresponding LLDB formatters."""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlDataFormatterSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect("frame variable empty_bitset", substrs=["size=0"])
        self.expect(
            "frame variable small_bitset",
            substrs=["size=13", "[2] = true", "[9] = true", "[11] = false"],
        )

        self.expect(
            "frame variable ili",
            substrs=["size=5", "[0] = 1", "[4] = 5"],
        )

        self.expect(
            "frame variable vec",
            substrs=["size=3", "[0] = 10", "[1] = 20", "[2] = 30"],
        )
        self.expect(
            "frame variable q",
            substrs=["size=3", "[0] = 10", "[2] = 30"],
        )
        self.expect(
            "frame variable st",
            substrs=["size=3", "[0] = 10", "[2] = 30"],
        )
        self.expect(
            "frame variable pq",
            substrs=["size=3", "[0] = 10", "[2] = 30"],
        )

        self.expect(
            "frame variable va",
            substrs=["size=4", "[0] = 1", "[3] = 1234"],
        )

        self.expect("frame variable ok", substrs=["Has Value=true", "Value = 7"])
        self.expect(
            "frame variable err",
            substrs=["Has Value=false", "Unexpected =", "boom"],
        )

        self.expect(
            "frame variable loc",
            substrs=['"main.cpp":6:1', '"int main()"'],
        )
        loc_empty = self.frame().FindVariable("loc_empty")
        self.assertTrue(loc_empty.GetError().Success())
        self.assertTrue(not loc_empty.summary)

        self.expect("frame variable ns", substrs=["ns = 1 ns"])
        self.expect("frame variable s", substrs=["s = 1234 s"])

        self.expect("frame variable ec", substrs=["value=2"])
        self.expect("frame variable p", substrs=["file.txt"])

        self.expect("frame variable it", substrs=["item = 3"])
