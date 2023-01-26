"""
Test that breakpoints (reason = breakpoint) have more priority than
plan completion (reason = step in/out/over) when reporting stop reason after step,
in particular 'step out' and 'step over', and in addition 'step in'.
Check for correct StopReason when stepping to the line with breakpoint,
which should be eStopReasonBreakpoint in general,
and eStopReasonPlanComplete when breakpoint's condition fails or it is disabled.
"""


import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ThreadPlanUserBreakpointsTestCase(TestBase):

    def setUp(self):
        TestBase.setUp(self)

        # Build and run to starting breakpoint
        self.build()
        src = lldb.SBFileSpec('main.cpp')
        (self.target, self.process, self.thread, _) = \
            lldbutil.run_to_source_breakpoint(self, '// Start from here', src)

        # Setup two more breakpoints
        self.breakpoints = [self.target.BreakpointCreateBySourceRegex('breakpoint_%i' % i, src)
            for i in range(2)]
        self.assertTrue(
            all(bp and bp.GetNumLocations() == 1 for bp in self.breakpoints),
            VALID_BREAKPOINT)

    def check_correct_stop_reason(self, breakpoint_idx, condition):
        self.assertState(self.process.GetState(), lldb.eStateStopped)
        if condition:
            # All breakpoints active, stop reason is breakpoint
            thread1 = lldbutil.get_one_thread_stopped_at_breakpoint(self.process, self.breakpoints[breakpoint_idx])
            self.assertEquals(self.thread, thread1, "Didn't stop at breakpoint %i." % breakpoint_idx)
        else:
            # Breakpoints are inactive, stop reason is plan complete
            self.assertEquals(self.thread.GetStopReason(), lldb.eStopReasonPlanComplete,
                'Expected stop reason to be step into/over/out for inactive breakpoint %i line.' % breakpoint_idx)

    def change_breakpoints(self, action):
        for bp in self.breakpoints:
            action(bp)

    def check_thread_plan_user_breakpoint(self, condition, set_up_breakpoint_func):
        # Make breakpoints active/inactive in different ways
        self.change_breakpoints(lambda bp: set_up_breakpoint_func(condition, bp))

        self.thread.StepInto()
        # We should be stopped at the breakpoint_0 line with the correct stop reason
        self.check_correct_stop_reason(0, condition)

        # This step-over creates a step-out from `func_1` plan
        self.thread.StepOver()
        # We should be stopped at the breakpoint_1 line with the correct stop reason
        self.check_correct_stop_reason(1, condition)

        # Check explicit step-out
        # Make sure we install the breakpoint at the right address:
        # step-out might stop on different lines, if the compiler
        # did or did not emit more instructions after the return
        return_addr = self.thread.GetFrameAtIndex(1).GetPC()
        step_out_breakpoint = self.target.BreakpointCreateByAddress(return_addr)
        self.assertTrue(step_out_breakpoint, VALID_BREAKPOINT)
        set_up_breakpoint_func(condition, step_out_breakpoint)
        self.breakpoints.append(step_out_breakpoint)
        self.thread.StepOut()
        # We should be stopped somewhere in the main frame with the correct stop reason
        self.check_correct_stop_reason(2, condition)

        # Run the process until termination
        self.process.Continue()
        self.assertState(self.process.GetState(), lldb.eStateExited)

    def set_up_breakpoints_condition(self, condition, bp):
        # Set breakpoint condition to true/false
        conditionStr = 'true' if condition else 'false'
        bp.SetCondition(conditionStr)

    def set_up_breakpoints_enable(self, condition, bp):
        # Enable/disable breakpoint
        bp.SetEnabled(condition)

    def set_up_breakpoints_callback(self, condition, bp):
        # Set breakpoint callback to return True/False
        bp.SetScriptCallbackBody('return %s' % condition)

    def test_thread_plan_user_breakpoint_conditional_active(self):
        # Test with breakpoints having true condition
        self.check_thread_plan_user_breakpoint(condition=True,
                                               set_up_breakpoint_func=self.set_up_breakpoints_condition)

    def test_thread_plan_user_breakpoint_conditional_inactive(self):
        # Test with breakpoints having false condition
        self.check_thread_plan_user_breakpoint(condition=False,
                                               set_up_breakpoint_func=self.set_up_breakpoints_condition)

    def test_thread_plan_user_breakpoint_unconditional_active(self):
        # Test with breakpoints enabled unconditionally
        self.check_thread_plan_user_breakpoint(condition=True,
                                               set_up_breakpoint_func=self.set_up_breakpoints_enable)

    def test_thread_plan_user_breakpoint_unconditional_inactive(self):
        # Test with breakpoints disabled unconditionally
        self.check_thread_plan_user_breakpoint(condition=False,
                                               set_up_breakpoint_func=self.set_up_breakpoints_enable)

    def test_thread_plan_user_breakpoint_callback_active(self):
        # Test with breakpoints with callback that returns 'True'
        self.check_thread_plan_user_breakpoint(condition=True,
                                               set_up_breakpoint_func=self.set_up_breakpoints_callback)

    def test_thread_plan_user_breakpoint_callback_inactive(self):
        # Test with breakpoints with callback that returns 'False'
        self.check_thread_plan_user_breakpoint(condition=False,
                                               set_up_breakpoint_func=self.set_up_breakpoints_callback)
