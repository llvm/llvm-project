"""
Test basic std::weak_ptr functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSharedPtr(TestBase):

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect_expr("w",
                         result_type="std::weak_ptr<int>",
                         result_summary="3 strong=1 weak=2",
                         result_children=[ValueCheck(name="__ptr_")])
        self.expect_expr("*w.lock()", result_type="element_type", result_value="3")
        self.expect_expr("*w.lock() = 5", result_type="element_type", result_value="5")
        self.expect_expr("*w.lock()", result_type="element_type", result_value="5")
        self.expect_expr("w.use_count()", result_type="long", result_value="1")
        self.expect("expr w.reset()")
        self.expect_expr("w.use_count()", result_type="long", result_value="0")
