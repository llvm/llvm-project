import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftExplicitModules(lldbtest.TestBase):

    @swiftTest
    def test_any_type(self):
        """Test explicit Swift modules"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect("expression c", substrs=['hello explicit'])
