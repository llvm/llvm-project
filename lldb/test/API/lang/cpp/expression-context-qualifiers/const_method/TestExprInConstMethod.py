import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        (_, process, _, _) = lldbutil.run_to_source_breakpoint(
            self, "Break: const_method begin", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr("bar()", result_value="2", result_type="int")
        self.expect(
            "expression m_const_mem = 2.0",
            error=True,
            substrs=[
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
            matching=False,
        )
        self.expect(
            "expression m_mem = 2.0",
            error=True,
            substrs=[
                "cannot assign to non-static data member within const member function",
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
        )
        self.expect_expr("m_mem", result_value="-2")

        # Test short and long --c++-ignore-context-qualifiers option.
        self.expect(
            "expression --c++-ignore-context-qualifiers -- m_mem = 3.0",
            error=False,
        )
        self.expect_expr("m_mem", result_value="3")

        self.expect(
            "expression -Q -- m_mem = 4.0",
            error=False,
        )
        self.expect_expr("m_mem", result_value="4")

        # Test --c++-ignore-context-qualifiers via SBExpressionOptions.
        options = lldb.SBExpressionOptions()
        options.SetBooleanLanguageOption("c++-ignore-context-qualifiers", True)
        self.expect_expr("m_mem = -2.0; m_mem", options=options, result_value="-2")

        self.expect_expr("((Foo*)this)->bar()", result_type="double", result_value="5")

        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "Break: const_method no-this lambda",
            lldb.SBFileSpec("main.cpp"),
        )

        self.expect(
            "expression x = 7.0",
            error=True,
            substrs=[
                "cannot assign to non-static data member within const member function",
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
        )
        self.expect_expr("x", result_value="2")

        self.expect_expr("x = -5; x", options=options, result_value="-5")

        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "Break: const_method mutable no-this lambda",
            lldb.SBFileSpec("main.cpp"),
        )

        self.expect_expr("x = 7.0; x", result_value="7")

        lldbutil.continue_to_source_breakpoint(
            self, process, "Break: const_method lambda", lldb.SBFileSpec("main.cpp")
        )

        # FIXME: mutating this capture should be disallowed in a non-mutable lambda.
        self.expect_expr("y = 8.0")
        self.expect_expr("bar()", result_value="2", result_type="int")
        self.expect(
            "expression m_const_mem = 2.0",
            error=True,
            substrs=[
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
            matching=False,
        )
        self.expect(
            "expression m_mem = 2.0",
            error=True,
            substrs=[
                "cannot assign to non-static data member within const member function",
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
        )
        self.expect_expr("m_mem", result_value="-2")

        self.expect_expr("m_mem = -1; m_mem", options=options, result_value="-1")

        self.expect_expr("((Foo*)this)->bar()", result_type="double", result_value="5")

        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "Break: const_method mutable lambda",
            lldb.SBFileSpec("main.cpp"),
        )

        self.expect_expr("y = 9.0")
        self.expect_expr("bar()", result_value="2", result_type="int")
        self.expect(
            "expression m_const_mem = 2.0",
            error=True,
            substrs=[
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
            matching=False,
        )
        self.expect(
            "expression m_mem = 2.0",
            error=True,
            substrs=[
                "cannot assign to non-static data member within const member function",
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
        )
        self.expect_expr("m_mem", result_value="-1")

        self.expect_expr("m_mem = -2; m_mem", options=options, result_value="-2")
