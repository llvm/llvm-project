import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftArrayUninitialized(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test unitialized global arrays"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect("target variable -- array_unused", substrs=['<uninitialized>'])
        self.expect("target variable -- array_used_empty", substrs=['0 values'])
