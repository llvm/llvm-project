import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftExpressionAccessControl(TestBase):

    @swiftTest
    def test_swift_expression_access_control(self):
        """Make sure expressions ignore access control"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))
        lldbutil.check_expression(self, self.frame(),
                                  "foo.m_a", "3", use_summary=False)

