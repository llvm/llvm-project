"""
Test that we can call functions and use types
annotated (and thus mangled) with ABI tags.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AbiTagLookupTestCase(TestBase):
    @skipIfWindows
    @expectedFailureAll(debug_info=["dwarf", "gmodules", "dwo"])
    def test_abi_tag_lookup(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp", False)
        )

        # Qualified/unqualified lookup to templates in namespace
        self.expect_expr("operator<(b1, b2)", result_type="bool", result_value="true")
        self.expect_expr(
            "A::operator<(b1, b2)", result_type="bool", result_value="true"
        )
        self.expect_expr("b1 < b2", result_type="bool", result_value="true")

        # Qualified/unqualified lookup to templates with ABI tags in namespace
        self.expect_expr("operator>(b1, b2)", result_type="bool", result_value="true")
        self.expect_expr(
            "A::operator>(b1, b2)", result_type="bool", result_value="true"
        )
        self.expect_expr("b1 > b2", result_type="bool", result_value="true")

        # Call non-operator templates with ABI tags
        self.expect_expr("A::withAbiTagInNS(1, 1)", result_type="int", result_value="1")

        self.expect_expr(
            "A::withAbiTagInNS(1.0, 1.0)", result_type="int", result_value="2"
        )
        self.expect_expr("withAbiTagInNS(b1, b2)", result_type="int", result_value="2")
        self.expect_expr(
            "A::withAbiTagInNS(b1, b2)", result_type="int", result_value="2"
        )

        self.expect_expr("withAbiTag(b1, b2)", result_type="int", result_value="3")
        self.expect_expr("withAbiTag(0, 0)", result_type="int", result_value="-3")

        # Structures with ABI tags
        self.expect_expr("t.Value()", result_type="const int", result_value="4")
        self.expect_expr("tt.Value()", result_type="const int", result_value="5")

        self.expect_expr(
            "Tagged{.mem = 6}",
            result_type="Tagged",
            result_children=[ValueCheck(name="mem", value="6")],
        )

        # Inline namespaces with ABI tags
        self.expect_expr(
            "v1::withImplicitTag(Simple{.mem = 6})", result_type="int", result_value="6"
        )
        self.expect_expr(
            "withImplicitTag(Simple{.mem = 6})", result_type="int", result_value="6"
        )

        self.expect_expr(
            "v1::withImplicitTag(Tagged{.mem = 6})", result_type="int", result_value="6"
        )
        self.expect_expr(
            "withImplicitTag(Tagged{.mem = 6})", result_type="int", result_value="6"
        )
