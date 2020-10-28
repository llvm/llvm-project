"""
Test basic std::vector functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBasicVector(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        vector_type = "std::vector<int, std::allocator<int> >"
        size_type = vector_type + "::size_type"
        value_type = "std::__vector_base<int, std::allocator<int> >::value_type"
        iterator = vector_type + "::iterator"
        # LLDB's formatter provides us with a artificial 'item' member.
        iterator_children = [ValueCheck(name="item")]
        riterator = vector_type + "::reverse_iterator"
        riterator_children = [
            ValueCheck(name="__t"),
            ValueCheck(name="current")
        ]

        self.expect_expr("a",
                         result_type=vector_type,
                         result_children=[
                             ValueCheck(value="3"),
                             ValueCheck(value="1"),
                             ValueCheck(value="2")
                         ])
        self.expect_expr("a.size()", result_type=size_type, result_value="3")
        self.expect_expr("a.front()", result_type=value_type, result_value="3")
        self.expect_expr("a[1]", result_type=value_type, result_value="1")
        self.expect_expr("a.back()", result_type=value_type, result_value="2")

        self.expect("expr std::sort(a.begin(), a.end())")
        self.expect_expr("a.front()", result_type=value_type, result_value="1")
        self.expect_expr("a[1]", result_type=value_type, result_value="2")
        self.expect_expr("a.back()", result_type=value_type, result_value="3")

        self.expect("expr std::reverse(a.begin(), a.end())")
        self.expect_expr("a.front()", result_type=value_type, result_value="3")
        self.expect_expr("a[1]", result_type=value_type, result_value="2")
        self.expect_expr("a.back()", result_type=value_type, result_value="1")

        self.expect_expr("a.begin()",
                         result_type=iterator,
                         result_children=iterator_children)
        self.expect_expr("a.rbegin()",
                         result_type=riterator,
                         result_children=riterator_children)

        self.expect_expr("*a.begin()", result_type="int", result_value="3")
        self.expect_expr("*a.rbegin()", result_type="int", result_value="1")

        self.expect("expr a.pop_back()")
        self.expect_expr("a.back()", result_type=value_type, result_value="2")
        self.expect_expr("a.size()", result_type=size_type, result_value="2")

        self.expect_expr("a.at(0)", result_type=value_type, result_value="3")

        self.expect("expr a.push_back(4)")
        self.expect_expr("a.back()", result_type=value_type, result_value="4")
        self.expect_expr("a.size()", result_type=size_type, result_value="3")

        self.expect("expr a.emplace_back(5)")
        self.expect_expr("a.back()", result_type=value_type, result_value="5")
        self.expect_expr("a.size()", result_type=size_type, result_value="4")

        self.expect_expr("a",
                         result_children=[
                             ValueCheck(value="3"),
                             ValueCheck(value="2"),
                             ValueCheck(value="4"),
                             ValueCheck(value="5")
                         ])
