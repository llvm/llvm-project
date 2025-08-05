"""
Test that writing memory does't affect hardware breakpoints.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from functionalities.breakpoint.hardware_breakpoints.base import *


class WriteMemoryWithHWBreakpoint(HardwareBreakpointTestBase):

    @skipTestIfFn(HardwareBreakpointTestBase.supports_hw_breakpoints)
    @skip
    def test_copy_memory_with_hw_break(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Run the program and stop at entry.
        self.expect(
            "process launch --stop-at-entry", patterns=["Process .* launched: .*a.out"]
        )

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)

        # Set a hardware breakpoint.
        bp_id = lldbutil.run_break_set_by_symbol(
            self, "hw_break_function", extra_options="--hardware"
        )

        # Get breakpoint location from the breakpoint.
        location = target.FindBreakpointByID(bp_id).GetLocationAtIndex(0)
        self.assertTrue(location and location.IsResolved(), VALID_BREAKPOINT_LOCATION)

        # Check that writing overlapping memory doesn't crash.
        address = location.GetLoadAddress()
        data = str("\x01\x02\x03\x04")
        error = lldb.SBError()

        result = process.WriteMemory(address, data, error)
        self.assertTrue(error.Success() and result == len(bytes))
