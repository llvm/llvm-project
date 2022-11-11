import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftNestedGenericClassTest(TestBase):

    @swiftTest
    def test(self):
        """Tests that a generic class type nested inside another generic class can be resolved correctly from the instance metadata"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"))
        self.expect("v foo", substrs=["a.A<Int>.B<String>"])

