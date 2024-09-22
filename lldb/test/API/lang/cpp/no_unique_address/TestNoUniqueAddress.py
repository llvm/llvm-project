"""
Test that LLDB correctly handles fields
marked with [[no_unique_address]].
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NoUniqueAddressTestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "return 0", lldb.SBFileSpec("main.cpp", False)
        )

        # Qualified/unqualified lookup to templates in namespace
        self.expect_expr(
            "b1",
            result_type="basic::Foo",
            result_children=[ValueCheck(name="a", type="Empty")],
        )

        self.expect_expr(
            "b2",
            result_type="bases::Foo",
            result_children=[
                ValueCheck(
                    type="bases::B", children=[ValueCheck(name="x", type="Empty")]
                ),
                ValueCheck(
                    type="bases::A",
                    children=[
                        ValueCheck(name="c", type="long", value="1"),
                        ValueCheck(name="d", type="long", value="2"),
                    ],
                ),
                ValueCheck(
                    type="bases::C", children=[ValueCheck(name="x", type="Empty")]
                ),
            ],
        )
        self.expect_expr(
            "b3",
            result_type="bases::Bar",
            result_children=[
                ValueCheck(
                    type="bases::B", children=[ValueCheck(name="x", type="Empty")]
                ),
                ValueCheck(
                    type="bases::C", children=[ValueCheck(name="x", type="Empty")]
                ),
                ValueCheck(
                    type="bases::A",
                    children=[
                        ValueCheck(name="c", type="long", value="5"),
                        ValueCheck(name="d", type="long", value="6"),
                    ],
                ),
            ],
        )

        self.expect("frame var b1")
        self.expect("frame var b2")
        self.expect("frame var b3")
