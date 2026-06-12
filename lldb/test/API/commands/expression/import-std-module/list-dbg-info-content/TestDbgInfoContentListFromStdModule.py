"""
Test basic std::list functionality but with a declaration from
the debug info (the Foo struct) as content.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDbgInfoContentList(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler="clang", compiler_version=["<", "12.0"])
    @skipIf(macos_version=["<", "15.0"])
    @skipIf(
        bugnumber="ASTImport of lambdas not supported: https://github.com/llvm/llvm-project/issues/149477"
    )
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.import-std-module true")

        if self.expectedCompiler(["clang"]) and self.expectedCompilerVersion(
            [">", "16.0"]
        ):
            list_type = "std::list<Foo>"
        else:
            list_type = "std::list<Foo, std::allocator<Foo> >"

        size_type = "size_type"
        value_type = "value_type"

        self.expect_expr(
            "a",
            result_type=list_type,
            result_children=[
                ValueCheck(children=[ValueCheck(value="3")]),
                ValueCheck(children=[ValueCheck(value="1")]),
                ValueCheck(children=[ValueCheck(value="2")]),
            ],
        )

        self.expect_expr("a.size()", result_type=size_type, result_value="3")
        self.expect_expr("a.front().a", result_type="int", result_value="3")
        self.expect_expr("a.back().a", result_type="int", result_value="2")

        self.expect("expr std::reverse(a.begin(), a.end())")
        self.expect_expr("a.front().a", result_type="int", result_value="2")
        self.expect_expr("a.back().a", result_type="int", result_value="3")

        self.expect_expr("a.begin()->a", result_type="int", result_value="2")
        self.expect_expr("a.rbegin()->a", result_type="int", result_value="3")
