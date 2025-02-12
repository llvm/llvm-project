from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftTypeAliasRecurtsive(TestBase):
    @swiftTest
    def test(self):
        """Test that type aliases of type aliases can be resolved"""
        self.build()
        self.runCmd("settings set symbols.swift-validate-typesystem false")
        self.expect("log enable lldb types")
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"))
        self.expect("frame variable cls", substrs=["ClassAlias?", "0x"])
