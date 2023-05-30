"""
Test that we correctly handle inline namespaces.
"""

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInlineNamespace(TestBase):
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        # The 'A::B::f' function must be found via 'A::f' as 'B' is an inline
        # namespace.
        self.expect_expr("A::f()", result_type="int", result_value="3")
        # But we should still find the function when we pretend the inline
        # namespace is not inline.
        self.expect_expr("A::B::f()", result_type="int", result_value="3")

        self.expect_expr("A::B::global_var", result_type="int", result_value="0")
        # FIXME: should be ambiguous lookup but ClangExpressionDeclMap takes
        #        first global variable that the lookup found, which in this case
        #        is A::B::global_var
        self.expect_expr("A::global_var", result_type="int", result_value="0")

        self.expect_expr("A::B::C::global_var", result_type="int", result_value="1")
        self.expect_expr("A::C::global_var", result_type="int", result_value="1")

        self.expect_expr("A::B::D::nested_var", result_type="int", result_value="2")
        self.expect_expr("A::D::nested_var", result_type="int", result_value="2")
        self.expect_expr("A::B::nested_var", result_type="int", result_value="2")
        self.expect_expr("A::nested_var", result_type="int", result_value="2")

        self.expect_expr("A::E::F::other_var", result_type="int", result_value="3")
        self.expect_expr("A::E::other_var", result_type="int", result_value="3")

        self.expect(
            "expr A::E::global_var",
            error=True,
            substrs=["no member named 'global_var' in namespace 'A::E'"],
        )
        self.expect(
            "expr A::E::F::global_var",
            error=True,
            substrs=["no member named 'global_var' in namespace 'A::E::F'"],
        )

        self.expect(
            "expr A::other_var",
            error=True,
            substrs=["no member named 'other_var' in namespace 'A'"],
        )
        self.expect(
            "expr A::B::other_var",
            error=True,
            substrs=["no member named 'other_var' in namespace 'A::B'"],
        )
        self.expect(
            "expr B::other_var",
            error=True,
            substrs=["no member named 'other_var' in namespace 'A::B'"],
        )

        # 'frame variable' can correctly distinguish between A::B::global_var and A::global_var
        gvars = self.target().FindGlobalVariables("A::global_var", 10)
        self.assertEqual(len(gvars), 1)
        self.assertEqual(gvars[0].GetValueAsSigned(), 4)

        self.expect("frame variable A::global_var", substrs=["(int) A::global_var = 4"])
        self.expect(
            "frame variable A::B::global_var", substrs=["(int) A::B::global_var = 0"]
        )
