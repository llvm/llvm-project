"""
Tests standard library iterators.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @expectedFailureAll(bugnumber="https://github.com/llvm/llvm-project/issues/149477")
    def test_xfail(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.import-std-module true")

        self.expect("expr move_begin++")
        self.expect_expr("move_begin + 2 == move_end", result_value="true")
        self.expect("expr move_begin--")
        self.expect_expr("move_begin + 3 == move_end", result_value="true")
