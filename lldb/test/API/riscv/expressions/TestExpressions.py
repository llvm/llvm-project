"""
Test RISC-V expressions evaluation.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestExpressions(TestBase):
    def common_setup(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_int_arg(self):
        self.common_setup()
        self.expect_expr("foo(foo(5), foo())", result_type="int", result_value="8")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_double_arg(self):
        self.common_setup()
        self.expect(
            "expr func_with_double_arg(1, 6.5)",
            error=True,
            substrs=["Architecture passes failure on function $__lldb_expr"],
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_ptr_arg(self):
        self.common_setup()
        self.expect(
            'expr func_with_ptr_arg("bla")',
            error=True,
            substrs=["Architecture passes failure on function $__lldb_expr"],
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_struct_arg(self):
        self.common_setup()
        self.expect_expr("func_with_struct_arg(s)", result_type="int", result_value="3")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_unsupported_struct_arg(self):
        self.common_setup()
        self.expect(
            "expr func_with_unsupported_struct_arg(u)",
            error=True,
            substrs=["Architecture passes failure on function $__lldb_expr"],
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_double_ret_val(self):
        self.common_setup()

        self.expect(
            "expr func_with_double_return()",
            error=True,
            substrs=["Architecture passes failure on function $__lldb_expr"],
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_ptr_ret_val(self):
        self.common_setup()
        self.expect(
            "expr func_with_ptr_return()",
            error=True,
            substrs=["Architecture passes failure on function $__lldb_expr"],
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_struct_return(self):
        self.common_setup()
        self.expect_expr("func_with_struct_return()", result_type="S")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_ptr_ret_val(self):
        self.common_setup()
        self.expect(
            "expr func_with_unsupported_struct_return()",
            error=True,
            substrs=["Architecture passes failure on function $__lldb_expr"],
        )
