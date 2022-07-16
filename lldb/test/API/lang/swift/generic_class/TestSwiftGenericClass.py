import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftGenericClassTest(TestBase):

    @swiftTest
    def test(self):
        """Tests that a generic class type can be resolved from the instance metadata alone"""
        self.build()
        (target, process, thread, breakpoint) = lldbutil.run_to_source_breakpoint(self, 
                "break here", lldb.SBFileSpec("main.swift"))

        self.expect("frame variable -d run self",
                    substrs=["a.F<Int>", "23", "42", "128", "256"])
        self.expect("expr -d run -- self",
                    substrs=["a.F<Int>", "23", "42", "128", "256"])
