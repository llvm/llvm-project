"""
Test that breakpoints set on a bad address say they are bad.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class BadAddressBreakpointTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_bad_address_breakpoints(self):
        """Test that breakpoints set on a bad address say they are bad."""
        self.build()
        self.address_breakpoints()

    def address_breakpoints(self):
        """Test that breakpoints set on a bad address say they are bad."""
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.c")
        )

        # illegal_address will hold (optionally) an address that, if
        # used as a breakpoint, will generate an unresolved breakpoint.
        illegal_address = None

        # Walk through all the memory regions in the process and
        # find an address that is invalid.
        regions = process.GetMemoryRegions()
        for region_idx in range(regions.GetSize()):
            region = lldb.SBMemoryRegionInfo()
            regions.GetMemoryRegionAtIndex(region_idx, region)
            if illegal_address == None or region.GetRegionEnd() > illegal_address:
                illegal_address = region.GetRegionEnd()

        if illegal_address is not None:
            # Now, set a breakpoint at the address we know is illegal.
            bkpt = target.BreakpointCreateByAddress(illegal_address)
            # Verify that breakpoint is not resolved.
            for bp_loc in bkpt:
                self.assertEqual(bp_loc.IsResolved(), False)
        else:
            self.fail(
                "Could not find an illegal address at which to set a bad breakpoint."
            )
