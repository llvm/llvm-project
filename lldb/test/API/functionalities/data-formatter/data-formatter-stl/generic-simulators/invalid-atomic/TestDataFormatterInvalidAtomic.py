"""
Test formatting of `std::atomic`s not from any STL
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class InvalidAtomicDataFormatterTestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr(
            "a",
            result_children=[
                ValueCheck(name="foo", value="1"),
                ValueCheck(name="bar", value="2"),
            ],
        )
        self.expect_expr(
            "b",
            result_children=[
                ValueCheck(name="foo", value="3"),
                ValueCheck(name="bar", value="4"),
            ],
        )

        self.expect_expr(
            "c",
            result_children=[
                ValueCheck(name="foo", value="5"),
                ValueCheck(name="bar", value="6"),
            ],
        )
        self.expect_expr(
            "d",
            result_children=[
                ValueCheck(name="foo", value="7"),
                ValueCheck(name="bar", value="8"),
            ],
        )
