"""
Test basic std::list functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBasicDeque(TestBase):

    @add_test_categories(["libc++"])
    @expectedFailureDarwin
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        deque_type = "std::deque<int>"
        size_type = "size_type"
        value_type = "value_type"
        iterator = "iterator"
        iterator_children = [
            ValueCheck(name="__m_iter_"),
            ValueCheck(name="__ptr_")
        ]
        riterator = "reverse_iterator"
        riterator_children = [
            ValueCheck(name="__t"),
            ValueCheck(name="current")
        ]

        self.expect_expr("a",
                         result_type=deque_type,
                         result_children=[
                             ValueCheck(value='3'),
                             ValueCheck(value='1'),
                             ValueCheck(value='2'),
                         ])

        self.expect_expr("a.size()", result_type=size_type, result_value="3")
        self.expect_expr("a.front()", result_type=value_type, result_value="3")
        self.expect_expr("a.back()", result_type=value_type, result_value="2")

        self.expect("expr std::sort(a.begin(), a.end())")
        self.expect_expr("a.front()", result_type=value_type, result_value="1")
        self.expect_expr("a.back()", result_type=value_type, result_value="3")

        self.expect("expr std::reverse(a.begin(), a.end())")
        self.expect_expr("a.front()", result_type=value_type, result_value="3")
        self.expect_expr("a.back()", result_type=value_type, result_value="1")

        self.expect_expr("*a.begin()", result_type="int", result_value="3")
        self.expect_expr("*a.rbegin()", result_type="int", result_value="1")
        self.expect_expr("a.begin()",
                         result_type=iterator,
                         result_children=iterator_children)
        self.expect_expr("a.rbegin()",
                         result_type=riterator,
                         result_children=riterator_children)
