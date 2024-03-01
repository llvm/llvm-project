"""Test Python APIs for setting, getting, and using address masks."""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AddressMasksTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_address_masks(self):
        self.build()
        (target, process, t, bp) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        process.SetAddressableBits(lldb.eAddressMaskTypeAll, 42)
        self.assertEqual(0x0000029500003F94, process.FixAddress(0x00265E9500003F94))

        # ~((1ULL<<42)-1) == 0xfffffc0000000000
        process.SetAddressMask(lldb.eAddressMaskTypeAll, 0xFFFFFC0000000000)
        self.assertEqual(0x0000029500003F94, process.FixAddress(0x00265E9500003F94))

        # Check that all bits can pass through unmodified
        process.SetAddressableBits(lldb.eAddressMaskTypeAll, 64)
        self.assertEqual(0x00265E9500003F94, process.FixAddress(0x00265E9500003F94))

        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 42, lldb.eAddressMaskRangeLow
        )
        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 15, lldb.eAddressMaskRangeHigh
        )
        self.assertEqual(0x000002950001F694, process.FixAddress(0x00265E950001F694))
        self.assertEqual(0xFFFFFFFFFFFFF694, process.FixAddress(0xFFA65E950000F694))

        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 42, lldb.eAddressMaskRangeAll
        )
        self.assertEqual(0x000002950001F694, process.FixAddress(0x00265E950001F694))
        self.assertEqual(0xFFFFFE950000F694, process.FixAddress(0xFFA65E950000F694))

        # Set a eAddressMaskTypeCode which has the low 3 bits marked as non-address
        # bits, confirm that they're cleared by FixAddress.
        process.SetAddressableBits(
            lldb.eAddressMaskTypeAll, 42, lldb.eAddressMaskRangeAll
        )
        mask = process.GetAddressMask(lldb.eAddressMaskTypeAny)
        process.SetAddressMask(lldb.eAddressMaskTypeCode, mask | 0x3)
        process.SetAddressMask(lldb.eAddressMaskTypeCode, 0xFFFFFC0000000003)
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
