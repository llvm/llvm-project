import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestBoundOpaqueArchetype(TestBase):

    @swiftTest
    @expectedFailureWindows
    def test(self):
        """Tests that a type bound to an opaque archetype can be resolved correctly"""         
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"))
        self.expect("v s", substrs=["a.S<a.C>"])
        self.expect("po s", substrs=["S<C>"])

