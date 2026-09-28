"""
Test that variables of type char are displayed correctly.
"""

import AbstractBase

from lldbsuite.test.decorators import *


class CharTypeTestCase(AbstractBase.GenericTester):
    TEST_WITH_PDB_DEBUG_INFO = True

    def test_char_type(self):
        """Test that char-type variables are displayed correctly."""
        self.build_and_run("char.cpp", ["char"], qd=True)

    @requireDarwin
    def test_char_type_from_block(self):
        """Test that char-type variables are displayed correctly from a block."""
        self.build_and_run("char.cpp", ["char"], bc=True, qd=True)

    def test_unsigned_char_type(self):
        """Test that 'unsigned_char'-type variables are displayed correctly."""
        self.build_and_run("unsigned_char.cpp", ["unsigned", "char"], qd=True)

    @requireDarwin
    def test_unsigned_char_type_from_block(self):
        """Test that 'unsigned char'-type variables are displayed correctly from a block."""
        self.build_and_run("unsigned_char.cpp", ["unsigned", "char"], bc=True, qd=True)
