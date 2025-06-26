import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @swiftTest
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect("frame var ints", substrs=["2 values", "[0] = 6", "[1] = 7"])
        self.expect(
            "frame var strings", substrs=["2 values", '[0] = "six"', '[1] = "seven"']
        )
        self.expect(
            "frame var things", substrs=["1 value", "[0] = (id = 67, odd = true)"]
        )
