"""
Tests a C++ fixit for the `expr` command and
`po` alias (aka DWIM aka "do what I mean") alias.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test_fixit_with_dwim(self):
        """Confirms `po` shows an expression after applying Fix-It(s)."""

        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "dwim-print -O -- class C { int i; void f() { []() { ++i; }(); } }; 42",
            error=True,
            substrs=[
                "Evaluated this expression after applying Fix-It(s)",
                "class C { int i; void f() { [this]() { ++i; }(); } }",
            ],
        )

    def test_fixit_with_expression(self):
        """Confirms `expression` shows an expression after applying Fix-It(s)."""

        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "expr class C { int i; void f() { []() { ++i; }(); } }; 42",
            error=True,
            substrs=[
                "Evaluated this expression after applying Fix-It(s)",
                "class C { int i; void f() { [this]() { ++i; }(); } }",
            ],
        )
