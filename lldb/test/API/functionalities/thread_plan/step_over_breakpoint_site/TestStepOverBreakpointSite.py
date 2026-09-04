"""
Test that a thread plan queued while the thread sits on a breakpoint site keeps
control of the process.

Resuming from a stop on an enabled breakpoint site makes lldb push a
ThreadPlanStepOverBreakpoint to single-step off the site first.  That plan
auto-continues, and the stop it produces must not consume the plans below it:
they keep their state, and the site the single step lands on is hit for real on
the resume.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class StepOverBreakpointSiteTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_run_to_address_from_a_breakpoint_site(self):
        """The pc is on an enabled breakpoint site when the plan is queued."""
        thread, target, process = self.setup_target()
        self.expect_run_to_next_instruction(target, process, thread)

    def test_run_to_address_off_a_breakpoint_site(self):
        """The control: the same plan, with the pc one instruction past the
        site, which needs no step-over plan at all."""
        thread, target, process = self.setup_target()
        thread.StepInstruction(False)
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.expect_run_to_next_instruction(target, process, thread)

    def test_user_breakpoint_at_the_target_is_still_reported(self):
        """A user breakpoint at the address the plan runs to must still be
        reported as a hit: the thread steps off the site it was stopped on and
        the site it lands on is hit on the resume, not swallowed."""
        thread, target, process = self.setup_target()
        next_pc = self.next_pc(target, thread)
        user_bp = target.BreakpointCreateByAddress(next_pc)
        self.assertTrue(user_bp.GetNumLocations() > 0, VALID_BREAKPOINT)

        thread.RunToAddress(next_pc)

        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), next_pc)
        self.assertEqual(
            user_bp.GetHitCount(),
            1,
            "the user breakpoint at the address run to was not reported as hit",
        )
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

    def test_run_to_address_two_instructions_from_a_breakpoint_site(self):
        """A target further than one instruction away must be reached, not
        merely stepped towards: the single step off the site lands short of it,
        and the thread has to carry on to the address it was given."""
        thread, target, process = self.setup_target()
        pc = thread.GetFrameAtIndex(0).GetPCAddress()
        instructions = target.ReadInstructions(pc, 3)
        self.assertEqual(instructions.GetSize(), 3)
        target_pc = (
            instructions.GetInstructionAtIndex(2).GetAddress().GetLoadAddress(target)
        )

        thread.RunToAddress(target_pc)

        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(
            thread.GetFrameAtIndex(0).GetPC(),
            target_pc,
            "the thread stopped short of the address the plan was given",
        )

    def test_the_stepped_over_breakpoint_is_hit_again(self):
        """The site the thread was parked on must be re-enabled after the plan
        that stepped off it is popped, so a later pass hits it again."""
        thread, target, process = self.setup_target()
        breakpoint = target.GetBreakpointAtIndex(0)
        self.assertEqual(breakpoint.GetHitCount(), 1)

        self.expect_run_to_next_instruction(target, process, thread)
        process.Continue()

        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(
            breakpoint.GetHitCount(),
            2,
            "the breakpoint that was stepped over was not hit on the next pass",
        )

    def test_scripted_plan_queueing_the_run_to_address(self):
        """The shape the issue reports: the run-to-address plan is queued by a
        scripted thread plan rather than by the API directly."""
        thread, target, process = self.setup_target()
        self.runCmd("command script import run_to_address_plan.py")
        next_pc = self.next_pc(target, thread)

        args = lldb.SBStructuredData()
        args.SetFromJSON('{"addr":%d}' % next_pc)
        err = thread.StepUsingScriptedThreadPlan(
            "run_to_address_plan.RunToAddress", args, True
        )
        self.assertSuccess(err)

        self.assertState(
            process.GetState(),
            lldb.eStateStopped,
            "the process ran away instead of stopping at the queued address",
        )
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), next_pc)

    def next_pc(self, target, thread):
        pc = thread.GetFrameAtIndex(0).GetPCAddress()
        instructions = target.ReadInstructions(pc, 2)
        self.assertEqual(
            instructions.GetSize(), 2, "could not read two instructions at the pc"
        )
        return instructions.GetInstructionAtIndex(1).GetAddress().GetLoadAddress(target)

    def expect_run_to_next_instruction(self, target, process, thread):
        next_pc = self.next_pc(target, thread)

        thread.RunToAddress(next_pc)

        self.assertState(
            process.GetState(),
            lldb.eStateStopped,
            "the process was not stopped at the address the plan was given",
        )
        self.assertEqual(
            thread.GetFrameAtIndex(0).GetPC(),
            next_pc,
            "the thread did not stop at the address the plan was given",
        )

    def setup_target(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.c")
        )
        return thread, target, process
