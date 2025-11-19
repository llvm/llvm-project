import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftExpressionActor(TestBase):
    @swiftTest
    def test_static_func(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here for static", lldb.SBFileSpec("main.swift")
        )

        self.expect("expr self", substrs=["A.Type"])

    @swiftTest
    def test_func(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here for func", lldb.SBFileSpec("main.swift")
        )

        self.expect("expr self", substrs=["(a.A)", "i = 42", 's = "Hello"'])
