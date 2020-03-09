"""
Test that variable expressions of type char are evaluated correctly.
"""

import AbstractBase

from lldbsuite.test.decorators import *


class CharTypeExprTestCase(AbstractBase.GenericTester):

    mydir = AbstractBase.GenericTester.compute_mydir(__file__)

    def test_char_type(self):
        """Test that char-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('char.cpp', set(['char']), qd=True)

    @skipUnlessDarwin
    def test_char_type_from_block(self):
        """Test that char-type variables are displayed correctly from a block."""
        self.build_and_run_expr('char.cpp', set(['char']), bc=True, qd=True)

    def test_unsigned_char_type(self):
        """Test that 'unsigned_char'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr(
            'unsigned_char.cpp', set(['unsigned', 'char']), qd=True)

    @skipUnlessDarwin
    def test_unsigned_char_type_from_block(self):
        """Test that 'unsigned char'-type variables are displayed correctly from a block."""
        self.build_and_run_expr(
            'unsigned_char.cpp', set(['unsigned', 'char']), bc=True, qd=True)
