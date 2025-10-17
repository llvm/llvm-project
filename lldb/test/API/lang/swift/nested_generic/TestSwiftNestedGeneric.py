import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftNestedGeneric(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test the inline array synthetic child provider and summary"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.runCmd("settings set symbols.swift-enable-ast-context false")
        self.expect(
            "frame variable v",
            substrs=[
                "HoldsNonNamespacedNestedStruct.NamespacedNestingStruct<Int>",
                "nested = 42",
            ],
        )
