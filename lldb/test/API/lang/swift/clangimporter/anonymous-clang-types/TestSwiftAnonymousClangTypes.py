import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftAnonymousClangTypes(lldbtest.TestBase):
    @swiftTest
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "frame variable twoStructs",
            substrs=[
                "(TwoAnonymousStructs) twoStructs = {",
                "= (x = 1, y = 2, z = 3)",
                "= (a = 4)",
            ],
        )

        self.expect(
            "frame variable twoUnions",
            substrs=[
                "(TwoAnonymousUnions) twoUnions = {",
                "   = {",
                "     = (x = 2)",
                "     = (y = 2, z = 3)",
                "  }",
                "   = {",
                "     = (a = 4, b = 5, c = 6)",
                "     = (d = 4, e = 5)",
            ],
        )
