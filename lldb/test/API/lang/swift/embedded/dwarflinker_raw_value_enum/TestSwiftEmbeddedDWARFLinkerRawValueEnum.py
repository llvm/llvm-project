"""
Test that dsymutil preserves the DW_TAG_enumerator children of an embedded
Swift raw-value enum.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedDWARFLinkerRawValueEnum(TestBase):
    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test(self):
        """A raw-value enum keeps its case names after dsymutil links the dSYM."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable ev",
            substrs=["Event", "fault"],
            error=False,
            matching=True,
        )
