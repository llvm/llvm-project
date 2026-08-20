"""
Check that when data is written over a software breakpoint site, it does not
corrupt the breakpoint instruction, and is later written to memory when the
breakpoint is removed.
"""

import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *


class WriteOverSoftwareBreakpoint(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # Could not find a way to make place_break_here visible to lldb on Windows.
    @skipIfWindows
    def test_write_over_breakpoint(self):
        TestBase.setUp(self)
        self.line = line_number("main.c", "// break here")
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True
        )
        self.runCmd("run", RUN_SUCCEEDED)
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        loop_start_breakpoint_addr = (
            target.breakpoints[0].GetLocationAtIndex(0).GetLoadAddress()
        )

        # Memory operations and breakpoint actions must be sent to the server
        # right away instead of waiting for the next continue event.
        self.runCmd("settings set target.process.disable-memory-cache on")
        self.runCmd("settings set target.process.use-delayed-breakpoints false")

        # At this point we are stopped at the start of the for loop.
        # We will set a further breakpoint in foo, and this is the one we will
        # test by writing over it then continuing to it.
        # We could use the same breakpoint, but arranging for dead code
        # immediately before it is more risky.

        # The breakpoint must be on an exact address, because we do not want
        # lldb to adjust it based on the debug information for prologues
        # and epilogues.
        symbol_contexts = target.FindSymbols("place_break_here")
        self.assertEqual(1, len(symbol_contexts))
        label_symbol = symbol_contexts[0].GetSymbol()
        self.assertTrue(label_symbol.IsValid())
        bkpt_address = label_symbol.GetStartAddress().GetLoadAddress(target)

        # lldb-server has its algorithm unit tested as part of
        # NativeProcessProtocol, but debugserver does not use that. So we will
        # do a few different types of overwrite here so we have coverage for both
        # debug servers.
        #
        # We cannot be sure what the software break size will be, so I'm assuming
        # 4 bytes because that's what Arm/AArch64 use. This means we may not be
        # getting full coverage on Thumb or x86 but the test should still pass
        # there.
        #
        # We assume that the instruction immediately before the breakpoint in
        # foo is dead code, so we are allowed to corrupt it.
        writes = [
            # Up to but not over breakpoint.
            (-4, 4),
            # Over start of breakpoint.
            (-2, 4),
            # Exactly over breakpoint.
            (0, 4),
            # Over end of breakpoint.
            (2, 4),
            # Immediately after breakpoint.
            (4, 4),
            # From before to after breakpoint.
            (-4, 12),
        ]

        for write_offset, write_size in writes:
            # Place a breakpoint immediately after the dead code in foo.
            bkpt = target.BreakpointCreateByAddress(bkpt_address)
            self.assertTrue(bkpt.IsValid())
            self.assertEqual(bkpt.GetNumLocations(), 1)
            self.assertFalse(bkpt.IsHardware())
            self.assertEqual(bkpt.GetLocationAtIndex(0).GetLoadAddress(), bkpt_address)

            check_address = bkpt_address + write_offset

            # Read around the breakpoint site. We assume that read subsitution
            # is working, so the data here is the original contents of memory
            # without the trap instruction.
            err = lldb.SBError()
            original_data = bytearray(
                process.ReadMemory(check_address, write_size, err)
            )
            self.assertSuccess(err)
            self.assertEqual(len(original_data), write_size)

            # Write around/in/over the breakpoint site.
            write_data = bytearray(range(write_size))
            wrote = process.WriteMemory(check_address, write_data, err)
            self.assertSuccess(err)
            self.assertEqual(wrote, write_size)

            # The data overlapping the breakpoint site should be in that breakpoint's
            # saved data. So it will appear as if all the data was written to memory,
            # even though it was not yet.
            after_write = bytearray(process.ReadMemory(check_address, write_size, err))
            self.assertSuccess(err)
            self.assertEqual(after_write, write_data)

            # The instruction in memory should still be intact so we can continue
            # to the breakpoint.
            process.Continue()

            thread = process.thread[0]
            self.assertState(process.GetState(), lldb.eStateStopped)
            self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
            # Should be stopped at the breakpoint we placed in foo. This proves that
            # the breakpoint instruction was intact.
            self.assertEqual(
                bkpt_address,
                thread.selected_frame.GetPC(),
            )

            # When the breakpoint is removed the saved bytes will be written to
            # memory.
            self.assertTrue(target.BreakpointDelete(bkpt.GetID()))

            data = process.ReadMemory(check_address, write_size, err)
            self.assertSuccess(err)
            self.assertEqual(bytearray(data), write_data)

            # Restore the original instruction data. The dead code before the
            # breakpoint is ok but the instructions after it must be put back
            # so we can continue.
            wrote = process.WriteMemory(check_address, original_data, err)
            self.assertSuccess(err)
            self.assertEqual(wrote, write_size)

            # Continue back to the start of the loop.
            process.Continue()
            self.assertState(process.GetState(), lldb.eStateStopped)
            self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
            self.assertEqual(loop_start_breakpoint_addr, thread.selected_frame.GetPC())
