"""
Test importing the 'std' C++ module and evaluate expressions with it.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ImportStdModule(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        # Activate importing of std module.
        self.runCmd("settings set target.import-std-module true")
        # Calling some normal std functions that return non-template types.
        self.expect_expr("std::abs(-42)", result_type="int", result_value="42")
        self.expect_expr(
            "std::minmax<int>(1, 2).first", result_type="const int", result_value="1"
        )
        self.expect_expr("std::div(2, 1).quot", result_type="int", result_value="2")
        # Using types from std.
        self.expect_expr(
            "(std::size_t)33U", result_type="std::size_t", result_value="33"
        )
        # Calling templated functions that return non-template types.
        self.expect_expr(
            "char char_a = 'b'; char char_b = 'a'; std::swap(char_a, char_b); char_a",
            result_type="char",
            result_value="'a'",
        )

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test_non_cpp_language(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        # Activate importing of std module.
        self.runCmd("settings set target.import-std-module true")
        # These languages don't support C++ modules, so they shouldn't
        # be able to evaluate the expression.
        self.expect("expr -l C -- std::minmax<int>(1, 2).first", error=True)
        self.expect("expr -l C99 -- std::minmax<int>(1, 2).first", error=True)
        self.expect("expr -l C11 -- std::minmax<int>(1, 2).first", error=True)
        self.expect("expr -l ObjC -- std::minmax<int>(1, 2).first", error=True)
