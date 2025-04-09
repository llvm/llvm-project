import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestExpressionOpenClass(TestBase):
    NO_DEBUG_INFO_TEST = True
    @swiftTest
    def test(self):
        """Tests calling an open function"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect("expr -- a.foo()", substrs=["23"])
