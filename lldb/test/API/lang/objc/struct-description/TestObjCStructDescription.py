import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.expect(
            "vo pair",
            substrs=["error:", "not a pointer type", "(Pair) pair = (f = 2, e = 3)"],
        )
        self.expect(
            "expr -O -- pair",
            substrs=["error:", "not a pointer type", "(Pair)  (f = 2, e = 3)"],
        )
