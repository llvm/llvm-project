"""
Test basic std::unique_ptr functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestUniquePtr(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect_expr(
            "s",
            result_type="std::unique_ptr<int, std::default_delete<int> >",
            result_summary="3",
            result_children=[ValueCheck(name="__value_")])
        self.expect_expr("*s", result_type="int", result_value="3")
        self.expect_expr("*s = 5", result_type="int", result_value="5")
        self.expect_expr("*s", result_type="int", result_value="5")
        self.expect_expr("(bool)s", result_type="bool", result_value="true")
        self.expect("expr s.reset()")
        self.expect_expr("(bool)s", result_type="bool", result_value="false")
