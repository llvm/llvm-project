"""
Test that a multi-payload enum's alignment is not fabricated from its
DW_AT_byte_size in embedded Swift.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedMultiPayloadEnumAlignment(TestBase):
    def setup_test(self):
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        return thread.GetSelectedFrame()

    @skipUnlessDarwin
    @skipUnlessEmbeddedSwift
    # The expected sizes and offsets assume Int64 is 8-byte aligned.
    @skipIf(archs=no_match(["arm64", "x86_64"]))
    @swiftTest
    def test_enum_stride(self):
        frame = self.setup_test()

        wide_enum = frame.FindVariable("w").GetType()
        self.assertIn("WideEnum", wide_enum.GetName())
        self.assertEqual(wide_enum.GetByteSize(), 34)

        pairs = frame.FindVariable("pairs")
        self.assertTrue(pairs.IsValid(), "pairs is valid")
        self.assertEqual(pairs.GetType().GetByteSize(), 74)
        self.assertEqual(pairs.GetNumChildren(), 2)

        base = pairs.GetLoadAddress()
        self.assertNotEqual(base, lldb.LLDB_INVALID_ADDRESS)
        self.assertEqual(pairs.GetChildAtIndex(0).GetLoadAddress() - base, 0)
        self.assertEqual(
            pairs.GetChildAtIndex(1).GetLoadAddress() - base,
            40,
            "WideEnum's stride is alignUp(34, 8) = 40, not alignUp(34, 34) = 66",
        )

        self.expect(
            "frame variable pairs",
            substrs=["first", "pair", "0 = 7", "1 = 8", "second", "0 = 9", "1 = 10"],
        )

    @skipUnlessDarwin
    @skipUnlessEmbeddedSwift
    @skipIf(archs=no_match(["arm64", "x86_64"]))
    @swiftTest
    def test_enum_alignment(self):
        frame = self.setup_test()

        prefixed = frame.FindVariable("prefixed")
        self.assertTrue(prefixed.IsValid(), "prefixed is valid")
        self.assertEqual(prefixed.GetType().GetByteSize(), 42)

        base = prefixed.GetLoadAddress()
        self.assertNotEqual(base, lldb.LLDB_INVALID_ADDRESS)
        payload = prefixed.GetChildMemberWithName("payload")
        self.assertTrue(payload.IsValid(), "payload is valid")
        self.assertEqual(
            payload.GetLoadAddress() - base,
            8,
            "WideEnum's alignment is 8 (the max of the payload alignments), not 34",
        )

        self.expect(
            "frame variable prefixed",
            substrs=["tag = 3", "payload", "pair", "0 = 11", "1 = 12"],
        )

    @skipUnlessDarwin
    @skipUnlessEmbeddedSwift
    @skipIf(archs=no_match(["arm64", "x86_64"]))
    @swiftTest
    def test_zero_sized_enum(self):
        frame = self.setup_test()

        zero = frame.FindVariable("zero")
        self.assertTrue(zero.IsValid(), "zero is valid")
        self.assertEqual(zero.GetType().GetByteSize(), 0)
        self.expect("frame variable zero", substrs=["only"])

        prefixed_zero = frame.FindVariable("prefixedZero")
        self.assertTrue(prefixed_zero.IsValid(), "prefixedZero is valid")
        self.assertEqual(prefixed_zero.GetType().GetByteSize(), 1)

        base = prefixed_zero.GetLoadAddress()
        self.assertNotEqual(base, lldb.LLDB_INVALID_ADDRESS)
        payload = prefixed_zero.GetChildMemberWithName("payload")
        self.assertTrue(payload.IsValid(), "payload is valid")
        self.assertEqual(
            payload.GetLoadAddress() - base,
            1,
            "a zero-sized enum is 1-byte aligned, so it follows the tag byte",
        )

        self.expect("frame variable prefixedZero", substrs=["tag = 4", "payload"])
