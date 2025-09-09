"""
Test std::vector functionality when it's contents are vectors.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestVectorOfVectors(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(macos_version=["<", "15.0"])
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        if self.expectedCompiler(["clang"]) and self.expectedCompilerVersion(
            [">", "16.0"]
        ):
            vector_type = "std::vector<int>"
            vector_of_vector_type = "std::vector<std::vector<int> >"
        else:
            vector_type = "std::vector<int>"
            vector_of_vector_type = (
                "std::vector<std::vector<int>, std::allocator<std::vector<int> > >"
            )

        size_type = "size_type"
        value_type = "value_type"

        self.runCmd("settings set target.import-std-module true")

        self.expect_expr(
            "a",
            result_type=vector_of_vector_type,
            result_children=[
                ValueCheck(
                    type=vector_type,
                    children=[
                        ValueCheck(value="1"),
                        ValueCheck(value="2"),
                        ValueCheck(value="3"),
                    ],
                ),
                ValueCheck(
                    type=vector_type,
                    children=[
                        ValueCheck(value="3"),
                        ValueCheck(value="2"),
                        ValueCheck(value="1"),
                    ],
                ),
            ],
        )
        self.expect_expr("a.size()", result_type=size_type, result_value="2")
        front = self.expect_expr(
            "a.front().front()", result_type=value_type, result_value="1"
        )
        self.expect_expr("a[1][1]", result_type=value_type, result_value="2")
        self.expect_expr("a.back().at(0)", result_type=value_type, result_value="3")
