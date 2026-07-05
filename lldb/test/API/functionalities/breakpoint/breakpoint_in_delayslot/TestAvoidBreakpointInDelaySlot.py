"""
Test specific to MIPS
"""

import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AvoidBreakpointInDelaySlotAPITestCase(TestBase):
    @skipIf(archs=no_match(re.compile("mips*")))
    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.expect("file " + exe, patterns=["Current executable set to .*a.out.*"])

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByName("main", "a.out")
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        list = target.FindFunctions("foo", lldb.eFunctionNameTypeAuto)
        self.assertEqual(list.GetSize(), 1)
        sc = list.GetContextAtIndex(0)
        self.assertEqual(sc.GetSymbol().GetName(), "foo")
        function = sc.GetFunction()
        self.assertTrue(function)
        self.function(function, target)

    def function(self, function, target):
        """Iterate over instructions in function and place a breakpoint on delay slot instruction"""
        # Get the list of all instructions in the function
        insts = function.GetInstructions(target)
        print(insts)
        i = 0
        for inst in insts:
            if inst.HasDelaySlot():
                # Remember the address of branch instruction.
                branchinstaddress = inst.GetAddress().GetLoadAddress(target)

                # Get next instruction i.e delay slot instruction.
                delayinst = insts.GetInstructionAtIndex(i + 1)
                delayinstaddr = delayinst.GetAddress().GetLoadAddress(target)

                # Set breakpoint on delay slot instruction
                breakpoint = target.BreakpointCreateByAddress(delayinstaddr)

                # Verify the breakpoint.
                self.assertTrue(
                    breakpoint and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
                )
                # Get the location from breakpoint
                location = breakpoint.GetLocationAtIndex(0)

                # Get the address where breakpoint is actually set.
                bpaddr = location.GetLoadAddress()

                # Breakpoint address should be adjusted to the address of
                # branch instruction.
                self.assertEqual(branchinstaddress, bpaddr)
                i += 1
            else:
                i += 1
