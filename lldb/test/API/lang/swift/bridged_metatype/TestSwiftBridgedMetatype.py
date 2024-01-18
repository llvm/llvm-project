"""
Test the formatting of briged Swift metatypes
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftBridgedMetatype(TestBase):
    @swiftTest
    @skipUnlessFoundation
    def test_swift_bridged_metatype(self):
        """Test the formatting of bridged Swift metatypes"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        var_k = self.frame().FindVariable("k")
        lldbutil.check_variable(self, var_k, False, "@thick NSString.Type")
