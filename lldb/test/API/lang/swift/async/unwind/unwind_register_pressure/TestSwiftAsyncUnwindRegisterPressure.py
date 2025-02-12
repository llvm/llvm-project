import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


# This test is a best effort attempt at creating a program with high register
# pressure while also making the linker avoid compact unwind.
class TestCase(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def set_breakpoints_all_instructions(self, target):
        funclet_name = "$s1a12ASYNC___1___SiyYaFTY0_"
        breakpoints = set()

        sym_ctx_list = target.FindFunctions(funclet_name)
        self.assertEqual(
            sym_ctx_list.GetSize(),
            1,
            f"failed to get symbol context for {funclet_name}",
        )
        function = sym_ctx_list[0].function

        instructions = list(function.GetInstructions(target))
        self.assertGreater(len(instructions), 0)
        for instruction in instructions:
            bp = target.BreakpointCreateBySBAddress(instruction.GetAddress())
            self.assertTrue(
                bp.IsValid(), f"failed to set bp inside funclet {funclet_name}"
            )
            breakpoints.add(bp.GetID())
        return breakpoints

    def check_unwind_ok(self, thread):
        # Check that we see the virtual backtrace:
        expected_funcnames = [
            "ASYNC___1___",
            "ASYNC___2___",
            "ASYNC___3___",
            "ASYNC___4___",
            "ASYNC___5___",
        ]
        frames = thread.frames
        self.assertGreater(
            len(frames), len(expected_funcnames), f"Invalid backtrace for {frames}"
        )
        actual_funcnames = [
            frame.GetFunctionName() for frame in frames[: len(expected_funcnames)]
        ]
        for expected_name, actual_name in zip(expected_funcnames, actual_funcnames):
            self.assertIn(expected_name, actual_name, f"Unexpected backtrace: {frames}")

    @swiftTest
    @skipIf(oslist=["windows", "linux"])
    def test(self):
        """Test that the debugger can unwind at all instructions of all funclets"""
        self.build()

        source_file = lldb.SBFileSpec("main.swift")
        target, process, _, bp = lldbutil.run_to_source_breakpoint(
            self, "BREAK HERE", source_file
        )
        target.DeleteAllBreakpoints()

        breakpoints = self.set_breakpoints_all_instructions(target)
        num_breakpoints = len(breakpoints)

        # Reach most breakpoints and ensure we can unwind in that position.
        while True:
            process.Continue()
            if process.GetState() == lldb.eStateExited:
                break
            thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
            self.assertTrue(thread.IsValid())
            bpid = thread.GetStopReasonDataAtIndex(0)
            breakpoints.remove(bpid)
            target.FindBreakpointByID(bpid).SetEnabled(False)

            self.check_unwind_ok(thread)

        # We will never hit all breakpoints we set, because of things like
        # overflow handling or other unreachable traps. However, it's good to
        # have some sanity check that we have hit at least a decent chunk of
        # them.
        breakpoints_not_hit = len(breakpoints)
        self.assertLess(breakpoints_not_hit / num_breakpoints, 0.25)
