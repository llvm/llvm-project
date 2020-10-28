"""
Test std::unique_ptr functionality with a decl from debug info as content.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestUniquePtrDbgInfoContent(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIfLinux # s.reset() causes link errors on ubuntu 18.04/Clang 9
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect_expr(
            "s",
            result_type="std::unique_ptr<Foo, std::default_delete<Foo> >",
            result_children=[ValueCheck(children=[ValueCheck(value="3")])])
        self.expect_expr("s->a", result_type="int", result_value="3")
        self.expect_expr("s->a = 5", result_type="int", result_value="5")
        self.expect_expr("(int)s->a", result_type="int", result_value="5")
        self.expect_expr("(bool)s", result_type="bool", result_value="true")
        self.expect("expr s.reset()")
        self.expect_expr("(bool)s", result_type="bool", result_value="false")
