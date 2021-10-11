"""
Test std::vector functionality when it's contents are vectors.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestVectorOfVectors(TestBase):

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        vector_type = "std::vector<int>"
        vector_of_vector_type = "std::vector<" + vector_type + " >"
        size_type = "size_type"
        value_type = "value_type"

        self.runCmd("settings set target.import-std-module true")

        self.expect_expr(
            "a",
            result_type=vector_of_vector_type,
            result_children=[
                ValueCheck(type="std::vector<int>",
                           children=[
                               ValueCheck(value='1'),
                               ValueCheck(value='2'),
                               ValueCheck(value='3'),
                           ]),
                ValueCheck(type="std::vector<int>",
                           children=[
                               ValueCheck(value='3'),
                               ValueCheck(value='2'),
                               ValueCheck(value='1'),
                           ]),
            ])
        self.expect_expr("a.size()", result_type=size_type, result_value="2")
        front = self.expect_expr("a.front().front()", result_type=value_type,
                                 result_value="1")
        self.expect_expr("a[1][1]", result_type=value_type, result_value="2")
        self.expect_expr("a.back().at(0)",
                         result_type=value_type,
                         result_value="3")
