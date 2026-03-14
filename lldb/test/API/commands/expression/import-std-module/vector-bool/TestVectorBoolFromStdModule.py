"""
Test basic std::vector<bool> functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBoolVector(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(bugnumber="rdar://100741983")
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        vector_type = "std::vector<bool>"
        size_type = "size_type"

        self.runCmd("settings set target.import-std-module true")

        self.expect_expr(
            "a",
            result_type=vector_type,
            result_children=[
                ValueCheck(value="false"),
                ValueCheck(value="true"),
                ValueCheck(value="false"),
                ValueCheck(value="true"),
            ],
        )
        self.expect_expr("a.size()", result_type=size_type, result_value="4")
        # FIXME: Without the casting the result can't be materialized.
        self.expect_expr("(bool)a.front()", result_type="bool", result_value="false")
        self.expect_expr("(bool)a[1]", result_type="bool", result_value="true")
        self.expect_expr("(bool)a.back()", result_type="bool", result_value="true")

        self.expect_expr("(bool)*a.begin()", result_type="bool", result_value="false")
        self.expect_expr("(bool)*a.rbegin()", result_type="bool", result_value="true")
