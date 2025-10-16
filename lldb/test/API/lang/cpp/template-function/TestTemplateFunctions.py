"""
Test that we can call C++ template fucntions.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TemplateFunctionsTestCase(TestBase):
    def do_test_template_function(self, add_cast):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        if add_cast:
            self.expect_expr("(int) foo(42)", result_type="int", result_value="42")
        else:
            self.expect(
                "expr b1 <=> b2",
                error=True,
                substrs=[
                    "warning:",
                    "'<=>' is a single token in C++20; add a space to avoid a change in behavior",
                ],
            )

            self.expect_expr("foo(42)", result_type="int", result_value="42")

            # overload with template case
            self.expect_expr("h(10)", result_type="int", result_value="10")

            # ADL lookup case
            self.expect_expr("f(A::C{})", result_type="int", result_value="4")

            # ADL lookup but no overload
            self.expect_expr("g(A::C{})", result_type="int", result_value="4")

            # variadic function cases
            self.expect_expr("var(1)", result_type="int", result_value="10")
            self.expect_expr("var(1, 2)", result_type="int", result_value="10")

            # Overloaded templated operator case
            self.expect_expr("b1 > b2", result_type="bool", result_value="true")
            self.expect_expr("b1 >> b2", result_type="bool", result_value="true")
            self.expect_expr("b1 << b2", result_type="bool", result_value="true")
            self.expect_expr("b1 == b2", result_type="bool", result_value="true")

            # Overloaded operator case
            self.expect_expr("d1 > d2", result_type="bool", result_value="true")
            self.expect_expr("d1 >> d2", result_type="bool", result_value="true")
            self.expect_expr("d1 << d2", result_type="bool", result_value="true")
            self.expect_expr("d1 == d2", result_type="bool", result_value="true")

    @skipIfWindows
    def test_template_function_with_cast(self):
        self.do_test_template_function(True)

    @skipIfWindows
    @expectedFailureAll(debug_info=["dwarf", "gmodules", "dwo"])
    def test_template_function_without_cast(self):
        self.do_test_template_function(False)
