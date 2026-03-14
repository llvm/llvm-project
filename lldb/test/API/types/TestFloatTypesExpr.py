"""
Test that variable expressions of floating point types are evaluated correctly.
"""


import AbstractBase

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FloatTypesExprTestCase(AbstractBase.GenericTester):
    # rdar://problem/8493023
    # test/types failures for Test*TypesExpr.py: element offset computed wrong
    # and sign error?

    def test_float_type(self):
        """Test that float-type variable expressions are evaluated correctly."""
        self.build_and_run_expr("float.cpp", set(["float"]))

    @skipUnlessDarwin
    def test_float_type_from_block(self):
        """Test that float-type variables are displayed correctly from a block."""
        self.build_and_run_expr("float.cpp", set(["float"]), bc=True)
