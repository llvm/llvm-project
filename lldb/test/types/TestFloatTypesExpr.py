"""
Test that variable expressions of floating point types are evaluated correctly.
"""

import AbstractBase
import unittest2
import lldb
import sys
from lldbtest import dsym_test, dwarf_test

class FloatTypesExprTestCase(AbstractBase.GenericTester):

    mydir = "types"

    # rdar://problem/8493023
    # test/types failures for Test*TypesExpr.py: element offset computed wrong and sign error?

    def setUp(self):
        # Call super's setUp().
        AbstractBase.GenericTester.setUp(self)
        # disable "There is a running process, kill it and restart?" prompt
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(lambda: self.runCmd("settings clear auto-confirm"))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_float_type_with_dsym(self):
        """Test that float-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('float.cpp', set(['float']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_float_type_from_block_with_dsym(self):
        """Test that float-type variables are displayed correctly from a block."""
        self.build_and_run_expr('float.cpp', set(['float']), bc=True)

    @dwarf_test
    def test_float_type_with_dwarf(self):
        """Test that float-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('float.cpp', set(['float']), dsym=False)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_double_type_with_dsym(self):
        """Test that double-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('double.cpp', set(['double']))

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_double_type_from_block_with_dsym(self):
        """Test that double-type variables are displayed correctly from a block."""
        self.build_and_run_expr('double.cpp', set(['double']), bc=True)

    @dwarf_test
    def test_double_type_with_dwarf(self):
        """Test that double-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('double.cpp', set(['double']), dsym=False)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
