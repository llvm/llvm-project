"""
Tests declaring RecordDecls in non-top-level expressions.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_struct(self):
        # Declare a struct and import it to the scratch AST.
        self.expect("expr struct S {}; S s; s", substrs=["= {}"])

    @no_debug_info_test
    def test_struct_with_fwd_decl_same_expr(self):
        # Test both a forward decl and a definition in one expression and
        # import them into the scratch AST.
        self.expect("expr struct S; struct S{}; S s; s", substrs=["= {}"])
