"""
Test basic std::pair functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect_expr("pair_int.first",
                         result_type="int",
                         result_value="1234")
        self.expect_expr("pair_int.second",
                         result_type="int",
                         result_value="5678")
        self.expect_expr("pair_int",
                         result_type="std::pair<int, int>",
                         result_children=[
                             ValueCheck(name="first", value="1234"),
                             ValueCheck(name="second", value="5678"),
                         ])
        self.expect_expr(
            "std::pair<long, long> lp; lp.first = 3333; lp.second = 2344; lp",
            result_type="std::pair<long, long>",
            result_children=[
                ValueCheck(name="first", value="3333"),
                ValueCheck(name="second", value="2344"),
            ])
