import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftTuple(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    def test(self):
        """Test the String formatter under adverse conditions"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        # FIXME: It would be even better if this were an error.
        self.expect("frame variable zero", substrs=['<uninitialized>'])
        self.expect("frame variable random", substrs=['cannot decode string'])
