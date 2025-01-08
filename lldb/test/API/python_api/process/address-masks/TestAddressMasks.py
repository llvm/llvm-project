"""Test Python APIs for setting, getting, and using address masks."""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AddressMasksTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def reset_all_masks(self, process):
        process.SetAddressMask(
            lldb.eAddressMaskTypeAll,
            lldb.LLDB_INVALID_ADDRESS_MASK,
            lldb.eAddressMaskRangeAll,
        )
        self.runCmd("settings set target.process.virtual-addressable-bits 0")
        self.runCmd("settings set target.process.highmem-virtual-addressable-bits 0")

    @skipIf(archs=["arm"])  # 32-bit arm ABI hardcodes Code mask, is 32-bit
    def test_address_masks(self):
        self.build()
        (target, process, t, bp) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        process.SetAddressableBits(lldb.eAddressMaskTypeAll, 42)
        self.assertEqual(0x0000029500003F94, process.FixAddress(0x00265E9500003F94))
        self.reset_all_masks(process)

        # ~((1ULL<<42)-1) == 0xfffffc0000000000
        process.SetAddressMask(lldb.eAddressMaskTypeAll, 0xFFFFFC0000000000)
        self.assertEqual(0x0000029500003F94, process.FixAddress(0x00265E9500003F94))
        self.reset_all_masks(process)

        # Check that all bits can pass through unmodified
        process.SetAddressableBits(lldb.eAddressMaskTypeAll, 64)
        self.assertEqual(0x00265E9500003F94, process.FixAddress(0x00265E9500003F94))
        self.reset_all_masks(process)

        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 42, lldb.eAddressMaskRangeAll
        )
        self.assertEqual(0x000002950001F694, process.FixAddress(0x00265E950001F694))
        self.assertEqual(0xFFFFFE950000F694, process.FixAddress(0xFFA65E950000F694))
        self.reset_all_masks(process)

        # Set a eAddressMaskTypeCode which has the low 3 bits marked as non-address
        # bits, confirm that they're cleared by FixAddress.
        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 42, lldb.eAddressMaskRangeAll
        )
        mask = process.GetAddressMask(lldb.eAddressMaskTypeAny)
        process.SetAddressMask(lldb.eAddressMaskTypeCode, mask | 0x3)
        self.assertEqual(0x000002950001F697, process.FixAddress(0x00265E950001F697))
        self.assertEqual(0xFFFFFE950000F697, process.FixAddress(0xFFA65E950000F697))
        self.assertEqual(
            0x000002950001F697,
            process.FixAddress(0x00265E950001F697, lldb.eAddressMaskTypeData),
        )
        self.assertEqual(
            0x000002950001F694,
            process.FixAddress(0x00265E950001F697, lldb.eAddressMaskTypeCode),
        )
        self.reset_all_masks(process)

        # The user can override whatever settings the Process thinks should be used.
        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 42, lldb.eAddressMaskRangeLow
        )
        self.runCmd("settings set target.process.virtual-addressable-bits 15")
        self.assertEqual(0x0000000000007694, process.FixAddress(0x00265E950001F694))
        self.assertEqual(0xFFFFFFFFFFFFF694, process.FixAddress(0xFFA65E950000F694))
        self.runCmd("settings set target.process.virtual-addressable-bits 0")
        self.assertEqual(0x000002950001F694, process.FixAddress(0x00265E950001F694))
        self.reset_all_masks(process)

    # AArch64 can have different address masks for high and low memory, when different
    # page tables are set up.
    @skipIf(archs=no_match(["arm64", "arm64e", "aarch64"]))
    @skipIf(archs=["arm"])  # 32-bit arm ABI hardcodes Code mask, is 32-bit
    def test_address_masks_target_supports_highmem_tests(self):
        self.build()
        (target, process, t, bp) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 42, lldb.eAddressMaskRangeLow
        )
        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 15, lldb.eAddressMaskRangeHigh
        )
        self.assertEqual(0x000002950001F694, process.FixAddress(0x00265E950001F694))
        self.assertEqual(0xFFFFFFFFFFFFF694, process.FixAddress(0xFFA65E950000F694))
        self.reset_all_masks(process)

        # The user can override whatever settings the Process thinks should be used.
        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 42, lldb.eAddressMaskRangeAll
        )
        self.runCmd("settings set target.process.virtual-addressable-bits 15")
        self.runCmd("settings set target.process.highmem-virtual-addressable-bits 15")
        self.assertEqual(0x0000000000007694, process.FixAddress(0x00265E950001F694))
        self.assertEqual(0xFFFFFFFFFFFFF694, process.FixAddress(0xFFA65E950000F694))
        self.runCmd("settings set target.process.virtual-addressable-bits 0")
        self.runCmd("settings set target.process.highmem-virtual-addressable-bits 0")
        self.assertEqual(0x000002950001F694, process.FixAddress(0x00265E950001F694))
        self.reset_all_masks(process)

    # On most targets where we have a single mask for all address range, confirm
    # that the high memory masks are ignored.
    @skipIf(archs=["arm64", "arm64e", "aarch64"])
    @skipIf(archs=["arm"])  # 32-bit arm ABI hardcodes Code mask, is 32-bit
    def test_address_masks_target_no_highmem(self):
        self.build()
        (target, process, t, bp) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 42, lldb.eAddressMaskRangeLow
        )
        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 15, lldb.eAddressMaskRangeHigh
        )
        self.assertEqual(0x000002950001F694, process.FixAddress(0x00265E950001F694))
        self.assertEqual(0xFFFFFE950000F694, process.FixAddress(0xFFA65E950000F694))
        self.runCmd("settings set target.process.virtual-addressable-bits 15")
        self.runCmd("settings set target.process.highmem-virtual-addressable-bits 42")
        self.assertEqual(0x0000000000007694, process.FixAddress(0x00265E950001F694))
        self.assertEqual(0xFFFFFFFFFFFFF694, process.FixAddress(0xFFA65E950000F694))
