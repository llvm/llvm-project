import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        (_, process, _, _) = lldbutil.run_to_source_breakpoint(
            self, "Break here: const", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr("bar()", result_type="double", result_value="5")
        self.expect_expr("const_volatile_method()")
        self.expect_expr("const_method()")
        self.expect(
            "expression volatile_method()",
            error=True,
            substrs=[
                "has type 'const Foo'",
                "but function is not marked const",
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
        )

        options = lldb.SBExpressionOptions()
        options.SetBooleanLanguageOption("c++-ignore-context-qualifiers", True)
        options.SetIgnoreBreakpoints(True)
        self.expect_expr("volatile_method()", options=options)
        self.expect(
            "expression --c++-ignore-context-qualifiers -- bar()",
            error=True,
            substrs=["call to member function 'bar' is ambiguous"],
        )

        lldbutil.continue_to_source_breakpoint(
            self, process, "Break here: volatile", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr(
            "bar()", result_type="const char *", result_summary='"volatile_bar"'
        )
        self.expect_expr("const_volatile_method()")
        self.expect(
            "expression const_method()",
            error=True,
            substrs=[
                "has type 'volatile Foo'",
                "but function is not marked volatile",
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
        )
        self.expect_expr("volatile_method()")

        self.expect_expr("const_method()", options=options)
        self.expect(
            "expression --c++-ignore-context-qualifiers -- bar()",
            error=True,
            substrs=["call to member function 'bar' is ambiguous"],
        )

        lldbutil.continue_to_source_breakpoint(
            self, process, "Break here: const volatile", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr("bar()", result_type="int", result_value="2")
        self.expect_expr("other_cv_method()")

        self.expect(
            "expression const_method()",
            error=True,
            substrs=[
                "has type 'const volatile Foo'",
                "but function is not marked const or volatile",
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
        )
        self.expect(
            "expression volatile_method()",
            error=True,
            substrs=[
                "has type 'const volatile Foo'",
                "but function is not marked const or volatile",
                "note: Possibly trying to mutate object in a const context. Try running the expression with",
            ],
        )

        self.expect_expr("const_method()", options=options)
        self.expect_expr("volatile_method()", options=options)
        self.expect(
            "expression --c++-ignore-context-qualifiers -- bar()",
            error=True,
            substrs=["call to member function 'bar' is ambiguous"],
        )
