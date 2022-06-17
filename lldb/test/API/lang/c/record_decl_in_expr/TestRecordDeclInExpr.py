"""
Tests declaring RecordDecls in non-top-level expressions.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    @no_debug_info_test
    def test_fwd_decl(self):
        # Declare a forward decl and import it to the scratch AST.
        self.expect_expr("struct S; S *s = nullptr; s", result_type="S *")

    @no_debug_info_test
    def test_struct(self):
        # Declare a struct and import it to the scratch AST.
        self.expect("expr struct S {}; S s; s", substrs=["= {}"])

    @no_debug_info_test
    def test_struct_with_fwd_decl(self):
        # Import the forward decl to the scratch AST.
        self.expect_expr("struct S; S *s = nullptr; s", result_type="S *")
        # Merge the definition into the scratch AST.
        self.expect("expr struct S {}; S s; s", substrs=["= {}"])

    @no_debug_info_test
    def test_struct_with_fwd_decl_same_expr(self):
        # Test both a forward decl and a definition in one expression and
        # import them into the scratch AST.
        self.expect("expr struct S; struct S{}; S s; s", substrs=["= {}"])
