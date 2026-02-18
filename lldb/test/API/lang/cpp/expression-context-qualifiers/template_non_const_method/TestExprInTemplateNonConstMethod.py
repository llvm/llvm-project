import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        (_, process, _, _) = lldbutil.run_to_source_breakpoint(
            self, "Break: non_const_method begin", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr("bar()", result_value="5", result_type="double")
        self.expect(
            "expression m_const_mem = 2.0",
            error=True,
            substrs=[
                "cannot assign to non-static data member",
                "with const-qualified type",
            ],
        )

        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "Break: non_const_method no-this lambda",
            lldb.SBFileSpec("main.cpp"),
        )

        self.expect(
            "expression x = 7.0",
            error=True,
            substrs=[
                "cannot assign to non-static data member within const member function",
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
                "expression --c++-ignore-context-qualifiers -- x = 7.0",
            ],
        )

        options = lldb.SBExpressionOptions()
        options.SetBooleanLanguageOption("c++-ignore-context-qualifiers", True)
        self.expect_expr("x = 6.0; x", options=options, result_value="6")

        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "Break: non_const_method mutable no-this lambda",
            lldb.SBFileSpec("main.cpp"),
        )

        self.expect_expr("x = 7.0; x", result_value="7")

        lldbutil.continue_to_source_breakpoint(
            self, process, "Break: non_const_method lambda", lldb.SBFileSpec("main.cpp")
        )

        # FIXME: mutating this capture should be disallowed in a non-mutable lambda.
        self.expect_expr("y = 8.0")
        self.expect_expr("bar()", result_value="5", result_type="double")
        self.expect(
            "expression m_const_mem = 2.0",
            error=True,
            substrs=[
                "cannot assign to non-static data member",
                "with const-qualified type",
            ],
        )
        self.expect_expr("m_mem = 2.0; m_mem", result_value="2")

        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "Break: non_const_method mutable lambda",
            lldb.SBFileSpec("main.cpp"),
        )

        self.expect_expr("y = 9.0")
        self.expect_expr("bar()", result_value="5", result_type="double")
        self.expect(
            "expression m_const_mem = 2.0",
            error=True,
            substrs=[
                "cannot assign to non-static data member",
                "with const-qualified type",
            ],
        )
        self.expect_expr("m_mem = 4.0; m_mem", result_value="4")
