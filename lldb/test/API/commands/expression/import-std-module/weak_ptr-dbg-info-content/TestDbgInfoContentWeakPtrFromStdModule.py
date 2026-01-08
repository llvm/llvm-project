"""
Test std::weak_ptr functionality with a decl from debug info as content.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDbgInfoContentWeakPtr(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler="clang", compiler_version=["<", "17.0"])
    @skipIf(macos_version=["<", "15.0"])
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.import-std-module true")

        self.expect_expr(
            "w",
            result_type="std::weak_ptr<Foo>",
            result_children=[ValueCheck(name="pointer")],
        )
        self.expect_expr("*w.lock()", result_type="element_type")
        self.expect_expr("w.lock()->a", result_type="int", result_value="3")
        self.expect_expr("w.lock()->a = 5", result_type="int", result_value="5")
        self.expect_expr("w.lock()->a", result_type="int", result_value="5")
        self.expect_expr("w.use_count()", result_type="long", result_value="1")
        self.expect("expr w.reset()")
        self.expect_expr("w.use_count()", result_type="long", result_value="0")
