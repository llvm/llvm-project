"""
Test that we handle breakpoints on consecutive instructions correctly.
"""



import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ConsecutiveBreakpointsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def prepare_test(self):
        self.build()

        (self.target, self.process, self.thread, bkpt) = lldbutil.run_to_source_breakpoint(
                self, "Set breakpoint here", lldb.SBFileSpec("main.cpp"))

        # Set breakpoint to the next instruction
        frame = self.thread.GetFrameAtIndex(0)

        address = frame.GetPCAddress()
        instructions = self.target.ReadInstructions(address, 2)
        self.assertTrue(len(instructions) == 2)
        self.bkpt_address = instructions[1].GetAddress()
        self.breakpoint2 = self.target.BreakpointCreateByAddress(
            self.bkpt_address.GetLoadAddress(self.target))
        self.assertTrue(
            self.breakpoint2 and self.breakpoint2.GetNumLocations() == 1,
            VALID_BREAKPOINT)

    def finish_test(self):
        # Run the process until termination
        self.process.Continue()
        self.assertEquals(self.process.GetState(), lldb.eStateExited)

    @no_debug_info_test
    def test_continue(self):
        """Test that continue stops at the second breakpoint."""
        self.prepare_test()

        self.process.Continue()
        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        # We should be stopped at the second breakpoint
        self.thread = lldbutil.get_one_thread_stopped_at_breakpoint(
            self.process, self.breakpoint2)
        self.assertIsNotNone(
            self.thread,
            "Expected one thread to be stopped at breakpoint 2")

        self.finish_test()

    @no_debug_info_test
    def test_single_step(self):
        """Test that single step stops at the second breakpoint."""
        self.prepare_test()

        step_over = False
        self.thread.StepInstruction(step_over)

        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        self.assertEquals(
            self.thread.GetFrameAtIndex(0).GetPCAddress().GetLoadAddress(
                self.target), self.bkpt_address.GetLoadAddress(
                self.target))
        self.thread = lldbutil.get_one_thread_stopped_at_breakpoint(
            self.process, self.breakpoint2)
        self.assertIsNotNone(
            self.thread,
            "Expected one thread to be stopped at breakpoint 2")

        self.finish_test()

    @no_debug_info_test
    def test_single_step_thread_specific(self):
        """Test that single step stops, even though the second breakpoint is not valid."""
        self.prepare_test()

        # Choose a thread other than the current one. A non-existing thread is
        # fine.
        thread_index = self.process.GetNumThreads() + 1
        self.assertFalse(self.process.GetThreadAtIndex(thread_index).IsValid())
        self.breakpoint2.SetThreadIndex(thread_index)

        step_over = False
        self.thread.StepInstruction(step_over)

        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        self.assertEquals(
            self.thread.GetFrameAtIndex(0).GetPCAddress().GetLoadAddress(
                self.target), self.bkpt_address.GetLoadAddress(
                self.target))
        self.assertEquals(
            self.thread.GetStopReason(),
            lldb.eStopReasonPlanComplete,
            "Stop reason should be 'plan complete'")

        self.finish_test()
