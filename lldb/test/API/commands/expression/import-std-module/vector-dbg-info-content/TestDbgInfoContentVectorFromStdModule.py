"""
Test basic std::vector functionality but with a declaration from
the debug info (the Foo struct) as content.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDbgInfoContentVector(TestBase):

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler="clang", compiler_version=['<', '12.0'])
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        if self.expectedCompilerVersion(['>', '16.0']):
            vector_type = "std::vector<Foo>"
        else:
            vector_type = "std::vector<Foo, std::allocator<Foo> >"

        size_type = "size_type"
        value_type = "value_type"
        iterator = "iterator"
        # LLDB's formatter provides us with a artificial 'item' member.
        iterator_children = [ValueCheck(name="item")]
        riterator = "reverse_iterator"
        riterator_children = [
            ValueCheck(), # Deprecated __t_ member; no need to check
            ValueCheck(name="current")
        ]

        self.expect_expr("a",
                         result_type=vector_type,
                         result_children=[
                             ValueCheck(children=[ValueCheck(value="3")]),
                             ValueCheck(children=[ValueCheck(value="1")]),
                             ValueCheck(children=[ValueCheck(value="2")]),
                         ])

        self.expect_expr("a.size()", result_type=size_type, result_value="3")
        self.expect_expr("a.front().a", result_type="int", result_value="3")
        self.expect_expr("a[1].a", result_type="int", result_value="1")
        self.expect_expr("a.back().a", result_type="int", result_value="2")

        self.expect("expr std::reverse(a.begin(), a.end())")
        self.expect_expr("a.front().a", result_type="int", result_value="2")

        self.expect_expr("a.begin()->a", result_type="int", result_value="2")
        self.expect_expr("a.rbegin()->a", result_type="int", result_value="3")

        self.expect("expr a.pop_back()")
        self.expect_expr("a.back().a", result_type="int", result_value="1")
        self.expect_expr("a.size()", result_type=size_type, result_value="2")

        self.expect_expr("a.at(0).a", result_type="int", result_value="2")

        self.expect("expr a.push_back({4})")
        self.expect_expr("a.back().a", result_type="int", result_value="4")
        self.expect_expr("a.size()", result_type=size_type, result_value="3")

        self.expect_expr("a.begin()",
                         result_type=iterator,
                         result_children=iterator_children)
        self.expect_expr("a.rbegin()",
                         result_type=riterator,
                         result_children=riterator_children)
