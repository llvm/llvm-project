import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftCUnion(lldbtest.TestBase):

    @swiftTest
    def test_c_unions(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect("target variable -- i", substrs=['42'])
        self.expect("target variable -- d", substrs=['23'])
