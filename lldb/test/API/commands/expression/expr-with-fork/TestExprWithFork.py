"""
Test that expressions that call functions which fork
can be evaluated successfully.

Fork events during expression evaluation are handled by RunThreadPlan,
which silently resumes them by default. The stop-on-fork option on
EvaluateExpressionOptions can be used to interrupt the expression on fork.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprWithForkTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # --- Basic expression evaluation across fork/vfork ---

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_fork(self):
        """Test that expression evaluation succeeds when the expression calls fork()."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr(
            "fork_and_return(42, false)", result_type="int", result_value="42"
        )

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_vfork(self):
        """Test that expression evaluation succeeds when the expression calls vfork()."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr(
            "fork_and_return(42, true)", result_type="int", result_value="42"
        )

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_fork_trap(self):
        """Test that expression evaluation handles a child process that triggers a SIGTRAP."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr(
            "fork_and_return_trap(42)", result_type="int", result_value="1"
        )

    # --- follow-fork-mode child override during expression evaluation ---

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_fork_follow_child(self):
        """Test that expression evaluation succeeds with follow-fork-mode child."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        original_pid = process.GetProcessID()
        self.runCmd("settings set target.process.follow-fork-mode child")

        # During expression evaluation, DidFork should override follow-fork-mode
        # to parent so the expression thread is not lost.
        self.expect_expr(
            "fork_and_return(42, false)", result_type="int", result_value="42"
        )

        # Verify we are still debugging the original process.
        self.assertEqual(process.GetProcessID(), original_pid)

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_vfork_follow_child(self):
        """Test that expression evaluation succeeds with vfork and follow-fork-mode child."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        original_pid = process.GetProcessID()
        self.runCmd("settings set target.process.follow-fork-mode child")

        self.expect_expr(
            "fork_and_return(42, true)", result_type="int", result_value="42"
        )

        self.assertEqual(process.GetProcessID(), original_pid)

    # --- stop-on-fork: fork interrupts expression immediately ---

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_fork_stop_on_fork(self):
        """Test that stop-on-fork interrupts expression evaluation on fork."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        options = lldb.SBExpressionOptions()
        options.SetStopOnFork(True)

        value = thread.GetSelectedFrame().EvaluateExpression(
            "fork_and_return(42, false)", options
        )

        # The expression should be interrupted due to the fork.
        self.assertTrue(value.GetError().Fail())

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_fork_stop_on_fork_process_state(self):
        """Test that process state is valid after stop-on-fork interrupts on fork."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        original_pid = process.GetProcessID()
        options = lldb.SBExpressionOptions()
        options.SetStopOnFork(True)

        thread.GetSelectedFrame().EvaluateExpression(
            "fork_and_return(42, false)", options
        )

        # After stop-on-fork interruption, the process should still be
        # the original and should be in a stopped state.
        self.assertEqual(process.GetProcessID(), original_pid)
        self.assertEqual(process.GetState(), lldb.eStateStopped)

    # --- stop-on-fork with vfork: deferred to vforkdone ---

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_vfork_stop_on_fork(self):
        """Test that stop-on-fork with vfork defers to vforkdone and interrupts."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        options = lldb.SBExpressionOptions()
        options.SetStopOnFork(True)

        value = thread.GetSelectedFrame().EvaluateExpression(
            "fork_and_return(42, true)", options
        )

        # The expression should be interrupted at vforkdone (deferred from
        # vfork) because stop-on-fork is set.
        self.assertTrue(value.GetError().Fail())

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_vfork_stop_on_fork_process_state(self):
        """Test that process state is clean after vfork stop-on-fork interruption.

        When stop-on-fork interrupts during vfork, the stop is deferred to
        vforkdone. At that point DidVForkDone has already restored software
        breakpoints and decremented m_vfork_in_progress_count, so the process
        should be in a fully functional state."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        original_pid = process.GetProcessID()
        options = lldb.SBExpressionOptions()
        options.SetStopOnFork(True)

        thread.GetSelectedFrame().EvaluateExpression(
            "fork_and_return(42, true)", options
        )

        # Process should still be the original and in stopped state.
        self.assertEqual(process.GetProcessID(), original_pid)
        self.assertEqual(process.GetState(), lldb.eStateStopped)

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_vfork_stop_on_fork_breakpoints_work(self):
        """Test that breakpoints are functional after vfork stop-on-fork.

        This is the key regression test for the degraded vfork state bug.
        After the deferred vfork stop-on-fork interruption, DidVForkDone
        should have re-enabled software breakpoints. Verify by evaluating
        another expression that would require functional breakpoints."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        options = lldb.SBExpressionOptions()
        options.SetStopOnFork(True)

        # First expression: vfork + stop-on-fork → interrupted at vforkdone.
        value = thread.GetSelectedFrame().EvaluateExpression(
            "fork_and_return(42, true)", options
        )
        self.assertTrue(value.GetError().Fail())

        # Second expression without stop-on-fork: should complete normally.
        # This verifies that breakpoints are working (expression evaluation
        # relies on internal breakpoints for function call returns).
        self.expect_expr("x", result_type="int", result_value="42")

    # --- stop-on-fork=false (default): no interruption ---

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_fork_stop_on_fork_false(self):
        """Test that stop-on-fork=false allows fork expression to complete."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        options = lldb.SBExpressionOptions()
        options.SetStopOnFork(False)

        value = thread.GetSelectedFrame().EvaluateExpression(
            "fork_and_return(42, false)", options
        )

        # With stop-on-fork disabled, the expression should complete normally.
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsSigned(), 42)

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_vfork_stop_on_fork_false(self):
        """Test that stop-on-fork=false allows vfork expression to complete."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        options = lldb.SBExpressionOptions()
        options.SetStopOnFork(False)

        value = thread.GetSelectedFrame().EvaluateExpression(
            "fork_and_return(42, true)", options
        )

        # With stop-on-fork disabled, vfork expression should complete normally.
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsSigned(), 42)

    # --- SBExpressionOptions stop-on-fork API ---

    def test_stop_on_fork_default(self):
        """Test that stop-on-fork defaults to false."""
        options = lldb.SBExpressionOptions()
        self.assertFalse(options.GetStopOnFork())

    def test_stop_on_fork_set_get(self):
        """Test that SetStopOnFork/GetStopOnFork round-trip correctly."""
        options = lldb.SBExpressionOptions()

        options.SetStopOnFork(True)
        self.assertTrue(options.GetStopOnFork())

        options.SetStopOnFork(False)
        self.assertFalse(options.GetStopOnFork())

    # --- stop-on-fork with follow-fork-mode child ---

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_fork_stop_on_fork_follow_child(self):
        """Test stop-on-fork + follow-child: expression interrupted, still on parent."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        original_pid = process.GetProcessID()
        self.runCmd("settings set target.process.follow-fork-mode child")

        options = lldb.SBExpressionOptions()
        options.SetStopOnFork(True)

        value = thread.GetSelectedFrame().EvaluateExpression(
            "fork_and_return(42, false)", options
        )

        # Expression should be interrupted by fork, and DidFork should have
        # overridden follow-fork-mode to parent during expression evaluation.
        self.assertTrue(value.GetError().Fail())
        self.assertEqual(process.GetProcessID(), original_pid)

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_with_vfork_stop_on_fork_follow_child(self):
        """Test stop-on-fork + vfork + follow-child: interrupted at vforkdone, still on parent."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        original_pid = process.GetProcessID()
        self.runCmd("settings set target.process.follow-fork-mode child")

        options = lldb.SBExpressionOptions()
        options.SetStopOnFork(True)

        value = thread.GetSelectedFrame().EvaluateExpression(
            "fork_and_return(42, true)", options
        )

        # Expression should be interrupted (deferred to vforkdone), and
        # DidVFork should have overridden follow-fork-mode to parent.
        self.assertTrue(value.GetError().Fail())
        self.assertEqual(process.GetProcessID(), original_pid)

    # --- concurrent fork on another thread during expression ---

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_concurrent_fork_other_thread(self):
        """Test that a fork on another thread during expression is handled correctly.

        The expression function releases a mutex, letting a helper thread
        fork, then waits for it to complete.  With SetStopOthers(false),
        the helper thread runs and forks while the expression is active.
        The fork must be handled as a normal (non-expression) fork so
        follow-fork-mode is respected."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        original_pid = process.GetProcessID()

        options = lldb.SBExpressionOptions()
        options.SetStopOthers(False)

        value = thread.GetSelectedFrame().EvaluateExpression(
            "expr_with_concurrent_fork()", options
        )

        # Expression should complete successfully.
        self.assertSuccess(value.GetError())

        # The return value is the child PID — must be > 0 (fork succeeded).
        child_pid = value.GetValueAsSigned()
        self.assertGreater(child_pid, 0, "fork() should have succeeded")

        # We should still be debugging the original parent process.
        self.assertEqual(process.GetProcessID(), original_pid)
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        # Breakpoints should still be functional.
        self.expect_expr("x", result_type="int", result_value="42")

    @skipIfWindows
    @add_test_categories(["fork"])
    def test_expr_concurrent_fork_other_thread_follow_child(self):
        """Test that follow-fork-mode child is overridden during expression evaluation.

        When follow-fork-mode is child and a non-expression thread forks while
        an expression is running, DidFork detects the process-wide expression
        state and overrides follow-fork-mode to parent, protecting the
        expression thread.  The expression should complete successfully and
        the process should remain the original parent."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        original_pid = process.GetProcessID()
        self.runCmd("settings set target.process.follow-fork-mode child")

        options = lldb.SBExpressionOptions()
        options.SetStopOthers(False)

        # Redirect debugger output to capture the async warning message
        # that DidFork emits when overriding follow-fork-mode.
        output_path = self.getBuildArtifact("debugger_output.txt")
        outfile = open(output_path, "w")
        self.dbg.SetOutputFileHandle(outfile, False)

        value = thread.GetSelectedFrame().EvaluateExpression(
            "expr_with_concurrent_fork()", options
        )

        self.dbg.GetOutputFile().Flush()
        outfile.close()
        self.dbg.SetOutputFileHandle(None, False)

        with open(output_path, "r") as f:
            debugger_output = f.read()

        # Expression should complete successfully because DidFork overrides
        # follow-fork-mode to parent when any expression is running.
        self.assertSuccess(value.GetError())

        # The return value is the child PID — must be > 0 (fork succeeded).
        child_pid = value.GetValueAsSigned()
        self.assertGreater(child_pid, 0, "fork() should have succeeded")

        # We should still be debugging the original parent process.
        self.assertEqual(process.GetProcessID(), original_pid)
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        # Breakpoints should still be functional.
        self.expect_expr("x", result_type="int", result_value="42")

        # Verify the user-facing warning about the follow-fork-mode override.
        self.assertIn(
            "follow-fork-mode",
            debugger_output,
            "expected override warning in debugger output",
        )
        self.assertIn(
            str(child_pid), debugger_output, "expected child PID in debugger output"
        )
        self.assertTrue(
            "stopped" in debugger_output or "running" in debugger_output,
            "expected child stop status in debugger output",
        )
