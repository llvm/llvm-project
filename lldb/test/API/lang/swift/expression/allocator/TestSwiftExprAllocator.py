import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftExprAllocator(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    def test_allocator_self(self):
        """Test expressions involving self in a allocating constructor. In an
        allocator, self is just a local variable, not being passed in, but
        should still be recognized as self."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect("expression x", substrs=['23'])
