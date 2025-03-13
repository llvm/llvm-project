import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

# This test creates 5 async frames:
# * ASYNC___1___, which is called by
# * ASYNC___2___, which is called by
# * ...
# * ASYNC___5___, which is called by
#
# The number of frames is important to exercise all possible unwind plans:
# * The top frame (ASYNC___1___) is created by just inspecting the registers as
# they are.
# * The plan for ASYNC___1___ -> ASYNC___2___ is created through
# `GetRuntimeUnwindPlan`, which is responsible for the transition from a real
# frame to a virtual frame.
# * The plan for ASYNC___2___ -> ASYNC___3___ is created through
# GetFollowAsyncContextUnwindPlan, which is responsible to create a virtual
# frame from another virtual frame.
# * The plan for ASYNC___3___ -> ASYNC___4___ is created through
# GetFollowAsyncContextUnwindPlan, but this time it follow the code path where
# `is_indirect = true` (see its implementation).
# * The plan for ASYNC___4___ -> ASYNC___5___ is created through the same code
# path as the previous one. However, it is the first time an unwind plan
# created from that path is used to create another unwind plan.


class TestCase(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def set_breakpoints_all_funclets(self, target):
        funclet_names = [
            "$s1a12ASYNC___1___4condS2i_tYaF",
            "$s1a12ASYNC___1___4condS2i_tYaFTY0_",
            "$s1a12ASYNC___1___4condS2i_tYaFTQ1_",
            "$s1a12ASYNC___1___4condS2i_tYaFTY2_",
            "$s1a12ASYNC___1___4condS2i_tYaFTQ3_",
            "$s1a12ASYNC___1___4condS2i_tYaFTY4_",
            "$s1a12ASYNC___1___4condS2i_tYaFTQ5_",
            "$s1a12ASYNC___1___4condS2i_tYaFTY6_",
        ]

        breakpoints = set()
        for funclet_name in funclet_names:
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

    unwind_fail_range_cache = dict()

    # There are challenges when unwinding Q funclets ("await resume"): LLDB cannot
    # detect the transition point where x22 stops containing the indirect context,
    # and instead contains the direct context.
    # Up to and including the first non-prologue instruction, LLDB correctly assumes
    # it is the indirect context.
    # After that, it assume x22 contains the direct context. There are a few
    # instructions where this is not true; this function computes a range that
    # includes such instructions, so the test may skip checks while stopped in them.
    def compute_unwind_fail_range(self, function, target):
        name = function.GetName()
        if name in TestCase.unwind_fail_range_cache:
            return TestCase.unwind_fail_range_cache[name]

        if "await resume" not in function.GetName():
            TestCase.unwind_fail_range_cache[name] = range(0)
            return range(0)

        first_pc_after_prologue = function.GetStartAddress()
        first_pc_after_prologue.OffsetAddress(function.GetPrologueByteSize())
        first_bad_instr = None
        first_good_instr = None
        for instr in function.GetInstructions(target):
            instr_addr = instr.GetAddress()

            # The first bad instruction is approximately the second instruction after the prologue
            # In actuality, it is at some point after that.
            if first_bad_instr is None and (
                instr_addr.GetFileAddress() > first_pc_after_prologue.GetFileAddress()
            ):
                first_bad_instr = instr
                continue

            # The first good instr is approximately the branch to swift_task_dealloc.
            # In actuality, it is at some point before that.
            if "swift_task_dealloc" in instr.GetComment(target):
                first_good_instr = instr
                break

            # If inside the bad range, no branches can be found.
            # If this happens, this test must fail so we know unwinding will be broken during stepping.
            if first_bad_instr is not None:
                # GetControlFlowKind is only implemented for x86.
                if "x86" in target.GetTriple():
                    self.assertEqual(
                        instr.GetControlFlowKind(target),
                        lldb.eInstructionControlFlowKindOther,
                        str(instr),
                    )

        self.assertNotEqual(first_bad_instr, None)
        self.assertNotEqual(first_good_instr, None)

        fail_range = range(
            first_bad_instr.GetAddress().GetFileAddress(),
            first_good_instr.GetAddress().GetFileAddress(),
        )
        TestCase.unwind_fail_range_cache[name] = fail_range
        return fail_range

    def should_skip_Q_funclet(self, thread):
        current_frame = thread.frames[0]
        function = current_frame.GetFunction()
        fail_range = self.compute_unwind_fail_range(
            function, thread.GetProcess().GetTarget()
        )

        current_pc = current_frame.GetPCAddress()
        return current_pc.GetFileAddress() in fail_range

    def check_unwind_ok(self, thread, bpid):
        if self.should_skip_Q_funclet(thread):
            return
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
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "BREAK HERE", source_file
        )

        breakpoints = self.set_breakpoints_all_funclets(target)
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

            self.check_unwind_ok(thread, bpid)

        # We will never hit all breakpoints we set, because of things like
        # overflow handling or other unreachable traps. However, it's good to
        # have some sanity check that we have hit at least a decent chunk of
        # them.
        breakpoints_not_hit = len(breakpoints)
        self.assertLess(breakpoints_not_hit / num_breakpoints, 0.10)
