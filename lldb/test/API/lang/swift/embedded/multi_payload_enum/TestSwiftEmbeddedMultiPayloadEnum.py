"""
Test that multi-payload enums are properly resolved in embedded Swift.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedMultiPayloadEnum(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test_multi_payload_enum(self):
        """Test that multi-payload enums display the correct case."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable multiSingle",
            substrs=["MultiPayloadEnum", "single", "42"],
        )

        self.expect(
            "frame variable multiPair",
            substrs=["MultiPayloadEnum", "pair", "1", "2.5"],
        )

        self.expect(
            "frame variable multiTriple",
            substrs=["MultiPayloadEnum", "triple", "1", "2.5", "true"],
        )

    @skipUnlessDarwin
    @swiftTest
    def test_same_size_payloads(self):
        """Test enums where cases have same-sized payloads."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable intPair",
            substrs=["SameSizePayloads", "intPair", "10", "20"],
        )

        self.expect(
            "frame variable doublePair",
            substrs=["SameSizePayloads", "doublePair", "1.5", "2.5"],
        )

    @skipUnlessDarwin
    @swiftTest
    def test_mixed_payload_enum(self):
        """Test enums mixing no-payload and various payload cases."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable mixedEmpty",
            substrs=["MixedPayloadEnum", "empty"],
        )

        self.expect(
            "frame variable mixedOne",
            substrs=["MixedPayloadEnum", "oneArg", "100"],
        )

        self.expect(
            "frame variable mixedTwo",
            substrs=["MixedPayloadEnum", "twoArgs", "100", "200.5"],
        )

        self.expect(
            "frame variable mixedThree",
            substrs=["MixedPayloadEnum", "threeArgs", "100", "200.5", "false"],
        )

    @skipUnlessDarwin
    @swiftTest
    def test_struct_payload_enum(self):
        """Test enums with struct payloads of different sizes."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable structSmall",
            substrs=["StructPayloadEnum", "small", "x = 1"],
        )

        self.expect(
            "frame variable structMedium",
            substrs=["StructPayloadEnum", "medium", "x = 1", "y = 2"],
        )

        self.expect(
            "frame variable structLarge",
            substrs=["StructPayloadEnum", "large", "x = 1", "y = 2", "z = 3"],
        )
