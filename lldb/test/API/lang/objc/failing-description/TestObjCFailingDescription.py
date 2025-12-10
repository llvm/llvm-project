import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.expect(
            "expr -O -- bad", substrs=["error:", "expression interrupted", "(Bad *) 0x"]
        )
        self.expect(
            "dwim-print -O -- bad",
            substrs=["error:", "expression interrupted", "_lookHere = NO"],
        )
