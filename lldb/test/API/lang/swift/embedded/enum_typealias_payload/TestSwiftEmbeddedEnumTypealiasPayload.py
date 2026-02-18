"""
Test that enums with typealiased payload types work in embedded Swift.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedEnumTypealiasPayload(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test enums with typealiased payload types."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable intCase",
            substrs=["AliasedPayloadEnum", "intCase", "42"],
        )

        self.expect(
            "frame variable doubleCase",
            substrs=["AliasedPayloadEnum", "doubleCase", "3.14"],
        )

        self.expect(
            "frame variable empty",
            substrs=["AliasedPayloadEnum", "empty"],
        )

        self.expect(
            "frame variable first",
            substrs=["NestedAliasedPayloadEnum", "first", "100"],
        )

        self.expect(
            "frame variable second",
            substrs=["NestedAliasedPayloadEnum", "second", "2.5"],
        )

        self.expect(
            "frame variable genericInt",
            substrs=["GenericAliasedPayloadEnum<Int>", "payload", "99"],
        )

        self.expect(
            "frame variable genericDouble",
            substrs=["GenericAliasedPayloadEnum<Double>", "payload", "1.5"],
        )

        self.expect(
            "frame variable genericEmpty",
            substrs=["GenericAliasedPayloadEnum<Int>", "empty"],
        )
