"""
Test thread plans queued while the thread sits on an enabled breakpoint site.
lldb first steps off the site with a ThreadPlanStepOverBreakpoint; that plan
auto-continues without consulting the plans already on the stack.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class StepOverBreakpointSiteTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def run_to_breakpoint(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.c")
        )
        return target, process, thread

    def pc_after(self, target, thread, count):
        """Return the load address `count` instructions after the current one."""
        pc = thread.GetFrameAtIndex(0).GetPCAddress()
        instructions = target.ReadInstructions(pc, count + 1)
        self.assertEqual(instructions.GetSize(), count + 1)
        return (
            instructions.GetInstructionAtIndex(count)
            .GetAddress()
            .GetLoadAddress(target)
        )

    def run_to_next_instruction(self, target, process, thread):
        next_pc = self.pc_after(target, thread, 1)
        thread.RunToAddress(next_pc)
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), next_pc)

    def test_run_to_address_from_a_breakpoint_site(self):
        """RunToAddress with the PC on a breakpoint site stops at the target."""
        target, process, thread = self.run_to_breakpoint()
        self.run_to_next_instruction(target, process, thread)

    def test_run_to_address_off_a_breakpoint_site(self):
        """RunToAddress with the PC off the breakpoint site stops at the target."""
        target, process, thread = self.run_to_breakpoint()
        thread.StepInstruction(False)
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.run_to_next_instruction(target, process, thread)

    def test_run_to_address_two_instructions_from_a_breakpoint_site(self):
        """RunToAddress reaches a target beyond the single step off the site."""
        target, process, thread = self.run_to_breakpoint()
        target_pc = self.pc_after(target, thread, 2)
        thread.RunToAddress(target_pc)
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), target_pc)

    def test_user_breakpoint_at_the_target_is_still_reported(self):
        """A user breakpoint at the RunToAddress target is reported as hit."""
        target, process, thread = self.run_to_breakpoint()
        next_pc = self.pc_after(target, thread, 1)
        user_bp = target.BreakpointCreateByAddress(next_pc)
        self.assertTrue(user_bp.GetNumLocations() > 0, VALID_BREAKPOINT)
        thread.RunToAddress(next_pc)
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), next_pc)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEqual(user_bp.GetHitCount(), 1)

    def test_the_stepped_over_breakpoint_is_hit_again(self):
        """The breakpoint stepped over before RunToAddress resumes is hit again."""
        target, process, thread = self.run_to_breakpoint()
        breakpoint = target.GetBreakpointAtIndex(0)
        self.assertEqual(breakpoint.GetHitCount(), 1)
        self.run_to_next_instruction(target, process, thread)
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEqual(breakpoint.GetHitCount(), 2)

    def test_scripted_plan_queueing_the_run_to_address(self):
        """A scripted plan that queues RunToAddress from a breakpoint site stops
        at the target."""
        target, process, thread = self.run_to_breakpoint()
        self.runCmd("command script import run_to_address_plan.py")
        next_pc = self.pc_after(target, thread, 1)
        args = lldb.SBStructuredData()
        args.SetFromJSON('{"addr":%d}' % next_pc)
        err = thread.StepUsingScriptedThreadPlan(
            "run_to_address_plan.RunToAddress", args, True
        )
        self.assertSuccess(err)
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), next_pc)
