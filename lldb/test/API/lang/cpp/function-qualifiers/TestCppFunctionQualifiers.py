import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestFunctionQualifiers(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Test calling a function that has const/non-const overload.
        self.expect_expr("c.func()", result_type="int", result_value="111")
        self.expect_expr("const_c.func()", result_type="int", result_value="222")

        # Call a function that is only const on a const/non-const instance.
        self.expect_expr("c.const_func()", result_type="int", result_value="333")
        self.expect_expr("const_c.const_func()", result_type="int", result_value="333")

        # Call a function that is not const on a const/non-const instance.
        self.expect_expr("c.nonconst_func()", result_type="int", result_value="444")
        self.expect(
            "expr const_c.nonconst_func()",
            error=True,
            substrs=[
                "'this' argument to member function 'nonconst_func' has type 'const C', but function is not marked const"
            ],
        )
