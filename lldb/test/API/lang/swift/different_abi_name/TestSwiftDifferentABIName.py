import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftDifferentABIName(TestBase):
    @swiftTest
    def test(self):
        self.build()

        _, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect("frame variable s", 
                    substrs=["Struct", "s = ", "field = 42"])
        self.expect("expr s", substrs=["Struct", "field = 42"])
        self.expect("expr -O -- s", substrs=["Struct", "- field : 42"])
