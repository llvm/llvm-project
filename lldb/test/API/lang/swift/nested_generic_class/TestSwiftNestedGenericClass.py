import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftNestedGenericClass(TestBase):
    @swiftTest
    def test(self):
        """Tests that a generic class type nested inside another generic class can be resolved correctly from the instance metadata"""
        self.build()
        _, process, _, breakpoint = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("v self", substrs=["A<Int>.B<String>"])

        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("v self", substrs=["C.D<Double>"])

        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("v self", substrs=["F.G<Bool>.H"])

        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("v self", substrs=["I.J<String, Int>"])

        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("v self", substrs=["K.L.M<Double, Bool>"])
