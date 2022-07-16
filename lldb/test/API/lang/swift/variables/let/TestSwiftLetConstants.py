import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftLetConstants(TestBase):

    @swiftTest
    def test_let_constants(self):
        """Test that let constants aren't writeable"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        # Test globals.
        self.expect("expression  -- global_constant = 3333", error=True)
        self.expect("frame variable global_constant", substrs=['1111'])
        self.expect("expression  -- global_variable = 4444") # FIXME:, substrs=['4444'])
        self.expect("frame variable global_variable", substrs=['4444'])

        # Test function parameters.
        self.expect("expression  -- parameter_constant = 3333", error=True)
        self.expect("frame variable parameter_constant", substrs=['1111'])
        self.expect("expression  -- parameter_variable = 4444") # FIXME:, substrs=['4444'])
        self.expect("frame variable parameter_variable", substrs=['4444'])

        # Test local variables.
        self.expect("expression  -- local_constant = 3333", error=True)
        self.expect("frame variable local_constant", substrs=['1111'])
        self.expect("expression  -- local_variable = 4444") # FIXME:, substrs=['4444'])
        self.expect("frame variable local_variable", substrs=['4444'])
