"""
Test basic std::list functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBasicList(TestBase):

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        if self.expectedCompilerVersion(['>', '16.0']):
            list_type = "std::list<int>"
        else:
            list_type = "std::list<int, std::allocator<int> >"

        size_type = "size_type"
        value_type = "value_type"

        iteratorvalue = "value_type"
        riterator_value = "value_type"

        self.expect_expr("a",
                         result_type=list_type,
                         result_children=[
                             ValueCheck(value="3"),
                             ValueCheck(value="1"),
                             ValueCheck(value="2")
                         ])

        self.expect_expr("a.size()", result_type=size_type, result_value="3")
        self.expect_expr("a.front()", result_type=value_type, result_value="3")
        self.expect_expr("a.back()", result_type=value_type, result_value="2")

        self.expect("expr a.sort()")
        self.expect_expr("a.front()", result_type=value_type, result_value="1")
        self.expect_expr("a.back()", result_type=value_type, result_value="3")

        self.expect("expr std::reverse(a.begin(), a.end())")
        self.expect_expr("a.front()", result_type=value_type, result_value="3")
        self.expect_expr("a.back()", result_type=value_type, result_value="1")

        self.expect_expr("*a.begin()",
                         result_type=iteratorvalue,
                         result_value="3")
        self.expect_expr("*a.rbegin()",
                         result_type=riterator_value,
                         result_value="1")
