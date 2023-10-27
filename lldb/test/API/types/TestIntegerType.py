"""
Test that variables of type integer are displayed correctly.
"""

import AbstractBase

from lldbsuite.test.decorators import *


class IntegerTypesTestCase(AbstractBase.GenericTester):
    def test_int_type(self):
        """Test that int-type variables are displayed correctly."""
        self.build_and_run("int.cpp", ["int"])

    @skipUnlessDarwin
    def test_int_type_from_block(self):
        """Test that int-type variables are displayed correctly from a block."""
        self.build_and_run("int.cpp", ["int"])

    def test_unsigned_int_type(self):
        """Test that 'unsigned_int'-type variables are displayed correctly."""
        self.build_and_run("unsigned_int.cpp", ["unsigned", "int"])

    @skipUnlessDarwin
    def test_unsigned_int_type_from_block(self):
        """Test that 'unsigned int'-type variables are displayed correctly from a block."""
        self.build_and_run("unsigned_int.cpp", ["unsigned", "int"], bc=True)
