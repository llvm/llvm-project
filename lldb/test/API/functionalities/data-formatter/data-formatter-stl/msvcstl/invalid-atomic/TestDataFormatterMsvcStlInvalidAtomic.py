"""
Test formatting of `std::atomic`s not from MSVC's STL
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlInvalidAtomicDataFormatterTestCase(TestBase):
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
