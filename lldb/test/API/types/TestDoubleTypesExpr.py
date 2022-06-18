"""
Test that variable expressions of floating point types are evaluated correctly.
"""



import AbstractBase

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DoubleTypesExprTestCase(AbstractBase.GenericTester):

    # rdar://problem/8493023
    # test/types failures for Test*TypesExpr.py: element offset computed wrong
    # and sign error?

    def test_double_type(self):
        """Test that double-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('double.cpp', set(['double']))

    @skipUnlessDarwin
    def test_double_type_from_block(self):
        """Test that double-type variables are displayed correctly from a block."""
        self.build_and_run_expr('double.cpp', set(['double']), bc=True)
