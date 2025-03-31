import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftGlobalOpaque(TestBase):

    @swiftTest
    def test(self):
        """Tests that a type bound to an opaque archetype can be resolved correctly"""         
        self.build()
        self.runCmd("settings set symbols.swift-enable-ast-context false")

        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"))

        self.expect("v v", substrs=["a.C", "i = 42"])
        self.expect("v s", substrs=["a.S<a.C>", "t = ", "i = 42"])


