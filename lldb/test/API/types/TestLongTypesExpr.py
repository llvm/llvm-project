"""
Test that variable expressions of integer basic types are evaluated correctly.
"""

import AbstractBase

from lldbsuite.test.decorators import *


class LongTypesExprTestCase(AbstractBase.GenericTester):

    mydir = AbstractBase.GenericTester.compute_mydir(__file__)

    def test_long_type(self):
        """Test that long-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('long.cpp', ['long'])

    @skipUnlessDarwin
    def test_long_type_from_block(self):
        """Test that long-type variables are displayed correctly from a block."""
        self.build_and_run_expr('long.cpp', ['long'], bc=True)

    def test_unsigned_long_type(self):
        """Test that 'unsigned long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_long.cpp', ['unsigned', 'long'])

    @skipUnlessDarwin
    def test_unsigned_long_type_from_block(self):
        """Test that 'unsigned_long'-type variables are displayed correctly from a block."""
        self.build_and_run_expr(
            'unsigned_long.cpp', ['unsigned', 'long'], bc=True)

    def test_long_long_type(self):
        """Test that 'long long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('long_long.cpp', ['long long'])

    @skipUnlessDarwin
    def test_long_long_type_from_block(self):
        """Test that 'long_long'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('long_long.cpp', ['long long'], bc=True)

    def test_unsigned_long_long_type(self):
        """Test that 'unsigned long long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_long_long.cpp',
                                ['unsigned', 'long long'])

    @skipUnlessDarwin
    def test_unsigned_long_long_type_from_block(self):
        """Test that 'unsigned_long_long'-type variables are displayed correctly from a block."""
        self.build_and_run_expr(
            'unsigned_long_long.cpp', ['unsigned', 'long long'], bc=True)
