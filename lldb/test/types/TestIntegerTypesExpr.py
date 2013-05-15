"""
Test that variable expressions of integer basic types are evaluated correctly.
"""

import AbstractBase
import unittest2
import lldb
import sys
from lldbtest import dsym_test, dwarf_test

class IntegerTypesExprTestCase(AbstractBase.GenericTester):

    mydir = "types"

    def setUp(self):
        # Call super's setUp().
        AbstractBase.GenericTester.setUp(self)
        # disable "There is a running process, kill it and restart?" prompt
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(lambda: self.runCmd("settings clear auto-confirm"))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_char_type_with_dsym(self):
        """Test that char-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('char.cpp', set(['char']), qd=True)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_char_type_from_block_with_dsym(self):
        """Test that char-type variables are displayed correctly from a block."""
        self.build_and_run_expr('char.cpp', set(['char']), bc=True, qd=True)

    @dwarf_test
    def test_char_type_with_dwarf(self):
        """Test that char-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('char.cpp', set(['char']), dsym=False, qd=True)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_char_type_with_dsym(self):
        """Test that 'unsigned_char'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_char.cpp', set(['unsigned', 'char']), qd=True)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_char_type_from_block_with_dsym(self):
        """Test that 'unsigned char'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('unsigned_char.cpp', set(['unsigned', 'char']), bc=True, qd=True)

    @dwarf_test
    def test_unsigned_char_type_with_dwarf(self):
        """Test that 'unsigned char'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_char.cpp', set(['unsigned', 'char']), dsym=False, qd=True)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_short_type_with_dsym(self):
        """Test that short-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('short.cpp', set(['short']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_short_type_from_block_with_dsym(self):
        """Test that short-type variables are displayed correctly from a block."""
        self.build_and_run_expr('short.cpp', set(['short']), bc=True)

    @dwarf_test
    def test_short_type_with_dwarf(self):
        """Test that short-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('short.cpp', set(['short']), dsym=False)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_short_type_with_dsym(self):
        """Test that 'unsigned_short'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_short.cpp', set(['unsigned', 'short']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_short_type_from_block_with_dsym(self):
        """Test that 'unsigned short'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('unsigned_short.cpp', set(['unsigned', 'short']), bc=True)

    @dwarf_test
    def test_unsigned_short_type_with_dwarf(self):
        """Test that 'unsigned short'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_short.cpp', set(['unsigned', 'short']), dsym=False)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_int_type_with_dsym(self):
        """Test that int-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('int.cpp', set(['int']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_int_type_from_block_with_dsym(self):
        """Test that int-type variables are displayed correctly from a block."""
        self.build_and_run_expr('int.cpp', set(['int']), dsym=False)

    @dwarf_test
    def test_int_type_with_dwarf(self):
        """Test that int-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('int.cpp', set(['int']), dsym=False)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_int_type_with_dsym(self):
        """Test that 'unsigned_int'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_int.cpp', set(['unsigned', 'int']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_int_type_from_block_with_dsym(self):
        """Test that 'unsigned int'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('unsigned_int.cpp', set(['unsigned', 'int']), bc=True)

    @dwarf_test
    def test_unsigned_int_type_with_dwarf(self):
        """Test that 'unsigned int'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_int.cpp', set(['unsigned', 'int']), dsym=False)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_long_type_with_dsym(self):
        """Test that long-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('long.cpp', set(['long']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_long_type_from_block_with_dsym(self):
        """Test that long-type variables are displayed correctly from a block."""
        self.build_and_run_expr('long.cpp', set(['long']), bc=True)

    @dwarf_test
    def test_long_type_with_dwarf(self):
        """Test that long-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('long.cpp', set(['long']), dsym=False)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_long_type_with_dsym(self):
        """Test that 'unsigned long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_long.cpp', set(['unsigned', 'long']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_long_type_from_block_with_dsym(self):
        """Test that 'unsigned_long'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('unsigned_long.cpp', set(['unsigned', 'long']), bc=True)

    @dwarf_test
    def test_unsigned_long_type_with_dwarf(self):
        """Test that 'unsigned long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_long.cpp', set(['unsigned', 'long']), dsym=False)

    # rdar://problem/8482903
    # test suite failure for types dir -- "long long" and "unsigned long long"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_long_long_type_with_dsym(self):
        """Test that 'long long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('long_long.cpp', set(['long long']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_long_long_type_from_block_with_dsym(self):
        """Test that 'long_long'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('long_long.cpp', set(['long long']), bc=True)

    @dwarf_test
    def test_long_long_type_with_dwarf(self):
        """Test that 'long long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('long_long.cpp', set(['long long']), dsym=False)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_long_long_type_with_dsym(self):
        """Test that 'unsigned long long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_long_long.cpp', set(['unsigned', 'long long']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_unsigned_long_long_type_from_block_with_dsym(self):
        """Test that 'unsigned_long_long'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('unsigned_long_long.cpp', set(['unsigned', 'long long']), bc=True)

    @dwarf_test
    def test_unsigned_long_long_type_with_dwarf(self):
        """Test that 'unsigned long long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_long_long.cpp', set(['unsigned', 'long long']), dsym=False)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
