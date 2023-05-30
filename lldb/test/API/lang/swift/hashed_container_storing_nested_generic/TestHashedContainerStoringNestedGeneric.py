import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftGenericClassInHashedContainerTest(TestBase):
    @swiftTest
    def test(self):
        """Tests that the type of hashed container whose value type is a non-generic class declared inside a generic class can be read from instance metadata"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("v val", substrs=["(a.A<Int>.B) val"])
