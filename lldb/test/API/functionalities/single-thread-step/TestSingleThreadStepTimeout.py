"""
Test that single thread step over deadlock issue can be resolved 
after timeout.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SingleThreadStepTimeoutTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.cpp"
        self.build()

    def verify_hit_correct_line(self, pattern):
        target_line = line_number(self.main_source, pattern)
        self.assertNotEqual(target_line, 0, "Could not find source pattern " + pattern)
        cur_line = self.thread.frames[0].GetLineEntry().GetLine()
        self.assertEqual(
            cur_line,
            target_line,
            "Stepped to line %d instead of expected %d with pattern '%s'."
            % (cur_line, target_line, pattern),
        )

    def step_over_deadlock_helper(self):
        (target, _, self.thread, _) = lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint1 here", lldb.SBFileSpec(self.main_source)
        )

        signal_main_thread_value = target.FindFirstGlobalVariable("signal_main_thread")
        self.assertTrue(signal_main_thread_value.IsValid())

        # Change signal_main_thread global variable to 1 so that worker thread loop can
        # terminate and move forward to signal main thread
        signal_main_thread_value.SetValueFromCString("1")

        self.thread.StepOver(lldb.eOnlyThisThread)
        self.verify_hit_correct_line("// Finish step-over from breakpoint1")

    @skipIfWindows
    def test_step_over_deadlock_small_timeout_fast_stepping(self):
        """Test single thread step over deadlock on other threads can be resolved after timeout with small timeout and fast stepping."""
        self.dbg.HandleCommand(
            "settings set target.process.thread.single-thread-plan-timeout 10"
        )
        self.dbg.HandleCommand("settings set target.use-fast-stepping true")
        self.step_over_deadlock_helper()

    @skipIfWindows
    def test_step_over_deadlock_small_timeout_slow_stepping(self):
        """Test single thread step over deadlock on other threads can be resolved after timeout with small timeout and slow stepping."""
        self.dbg.HandleCommand(
            "settings set target.process.thread.single-thread-plan-timeout 10"
        )
        self.dbg.HandleCommand("settings set target.use-fast-stepping false")
        self.step_over_deadlock_helper()

    @skipIfWindows
    def test_step_over_deadlock_large_timeout_fast_stepping(self):
        """Test single thread step over deadlock on other threads can be resolved after timeout with large timeout and fast stepping."""
        self.dbg.HandleCommand(
            "settings set target.process.thread.single-thread-plan-timeout 2000"
        )
        self.dbg.HandleCommand("settings set target.use-fast-stepping true")
        self.step_over_deadlock_helper()

    @skipIfWindows
    def test_step_over_deadlock_large_timeout_slow_stepping(self):
        """Test single thread step over deadlock on other threads can be resolved after timeout with large timeout and slow stepping."""
        self.dbg.HandleCommand(
            "settings set target.process.thread.single-thread-plan-timeout 2000"
        )
        self.dbg.HandleCommand("settings set target.use-fast-stepping false")
        self.step_over_deadlock_helper()

    def step_over_multi_calls_helper(self):
        (target, _, self.thread, _) = lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint2 here", lldb.SBFileSpec(self.main_source)
        )
        self.thread.StepOver(lldb.eOnlyThisThread)
        self.verify_hit_correct_line("// Finish step-over from breakpoint2")

    @skipIfWindows
    def test_step_over_multi_calls_small_timeout_fast_stepping(self):
        """Test step over source line with multiple call instructions works fine with small timeout and fast stepping."""
        self.dbg.HandleCommand(
            "settings set target.process.thread.single-thread-plan-timeout 10"
        )
        self.dbg.HandleCommand("settings set target.use-fast-stepping true")
        self.step_over_multi_calls_helper()

    @skipIfWindows
    def test_step_over_multi_calls_small_timeout_slow_stepping(self):
        """Test step over source line with multiple call instructions works fine with small timeout and slow stepping."""
        self.dbg.HandleCommand(
            "settings set target.process.thread.single-thread-plan-timeout 10"
        )
        self.dbg.HandleCommand("settings set target.use-fast-stepping false")
        self.step_over_multi_calls_helper()

    @skipIfWindows
    def test_step_over_multi_calls_large_timeout_fast_stepping(self):
        """Test step over source line with multiple call instructions works fine with large timeout and fast stepping."""
        self.dbg.HandleCommand(
            "settings set target.process.thread.single-thread-plan-timeout 2000"
        )
        self.dbg.HandleCommand("settings set target.use-fast-stepping true")
        self.step_over_multi_calls_helper()

    @skipIfWindows
    def test_step_over_multi_calls_large_timeout_slow_stepping(self):
        """Test step over source line with multiple call instructions works fine with large timeout and slow stepping."""
        self.dbg.HandleCommand(
            "settings set target.process.thread.single-thread-plan-timeout 2000"
        )
        self.dbg.HandleCommand("settings set target.use-fast-stepping false")
        self.step_over_multi_calls_helper()
