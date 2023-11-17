import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestBoundOpaqueArchetype(TestBase):

    @swiftTest
    def test(self):
        """Tests that a type bound to an opaque archetype can be resolved correctly"""         
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"))
        self.expect("v s", substrs=["S<", "opaque return type of", "f()"])
        self.expect("po s", substrs=["S<C>"])

