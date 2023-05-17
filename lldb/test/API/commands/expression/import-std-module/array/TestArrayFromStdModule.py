"""
Test basic std::array functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        size_type = "size_type"
        value_type = "value_type"
        iterator = "iterator"
        riterator = "reverse_iterator"

        # Test inspecting an array of integers.
        array_type = "std::array<int, 3>"
        self.expect_expr("a",
                         result_type=array_type,
                         result_children=[
                             ValueCheck(name="__elems_", children=[
                                 ValueCheck(value="3"),
                                 ValueCheck(value="1"),
                                 ValueCheck(value="2"),
                             ])
                         ])
        self.expect_expr("a.size()", result_type=size_type, result_value="3")
        self.expect_expr("a.front()", result_type=value_type, result_value="3")
        self.expect_expr("a[1]", result_type=value_type, result_value="1")
        self.expect_expr("a.back()", result_type=value_type, result_value="2")

        # Both are just pointers to the underlying elements.
        self.expect_expr("a.begin()", result_type=iterator)
        self.expect_expr("a.rbegin()", result_type=riterator)

        self.expect_expr("*a.begin()", result_type=value_type, result_value="3")
        self.expect_expr("*a.rbegin()", result_type="int", result_value="2")

        self.expect_expr("a.at(0)", result_type=value_type, result_value="3")


        # Same again with an array that has an element type from debug info.
        array_type = "std::array<DbgInfo, 1>"

        dbg_info_elem_children = [ValueCheck(value="4")]
        dbg_info_elem = [ValueCheck(children=dbg_info_elem_children)]

        self.expect_expr("b",
                         result_type=array_type,
                         result_children=[
                             ValueCheck(name="__elems_", children=dbg_info_elem)
                         ])
        self.expect_expr("b.size()", result_type=size_type, result_value="1")
        self.expect_expr("b.front()", result_type=value_type, result_children=dbg_info_elem_children)
        self.expect_expr("b[0]", result_type=value_type, result_children=dbg_info_elem_children)
        self.expect_expr("b.back()", result_type=value_type, result_children=dbg_info_elem_children)

        # Both are just pointers to the underlying elements.
        self.expect_expr("b.begin()", result_type=iterator)
        self.expect_expr("b.rbegin()", result_type=riterator)

        self.expect_expr("*b.begin()", result_type=value_type, result_children=dbg_info_elem_children)
        self.expect_expr("*b.rbegin()", result_type="DbgInfo", result_children=dbg_info_elem_children)

        self.expect_expr("b.at(0)", result_type=value_type, result_children=dbg_info_elem_children)

