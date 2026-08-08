"""
Test reading virtual bases through the vtable.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCppVirtualBases(TestBase):
    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        children = [
            ValueCheck(
                type="Padding4",
                children=[ValueCheck(type="short", name="member", value="4")],
            ),
            ValueCheck(
                type="User",
                children=[
                    ValueCheck(
                        type="Padding3",
                        children=[ValueCheck(type="short", name="member", value="3")],
                    ),
                    ValueCheck(
                        type="VBase1",
                        children=[ValueCheck(type="short", name="member", value="1")],
                    ),
                    ValueCheck(
                        type="VBase2",
                        children=[ValueCheck(type="short", name="member", value="2")],
                    ),
                    ValueCheck(type="short", name="member", value="6"),
                ],
            ),
            ValueCheck(
                type="Padding5",
                children=[ValueCheck(type="short", name="member", value="5")],
            ),
            ValueCheck(type="short", name="member", value="7"),
        ]
        self.expect_expr("useruser", result_type="UserUser", result_children=children)
        self.expect_var_path("useruser", type="UserUser", children=children)
