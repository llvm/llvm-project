"""
Test swift Class types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftActorTypes(TestBase):
    @swiftTest
    def test_swift_class_types(self):
        """Test swift Actor types"""
        self.build()
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('v actor', substrs=['Actor', 'str = "Hello"'])
        self.expect('expr actor', substrs=['Actor', 'str = "Hello"'])

