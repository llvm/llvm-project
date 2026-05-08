import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AlwaysRunThreadNamesTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    def test_always_run_thread_resumes_during_step(self):
        """Test that a thread named in always-run-thread-names continues
        running when another thread single-steps."""
        self.build()
        (target, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Configure the setting to keep our helper thread running.
        self.runCmd("settings set target.process.always-run-thread-names always-run")

        # Tell step_over_me() to block until the helper thread advances.
        self.runCmd("expression g_sync_with_helper = true")

        # Record the helper thread's counter before stepping.
        counter_before = target.FindFirstGlobalVariable("g_helper_count")
        self.assertTrue(counter_before.IsValid())
        val_before = counter_before.GetValueAsUnsigned()

        # The step over normally suspends all other threads, but the
        # always-run-thread-names setting keeps the helper thread running.
        # Because g_sync_with_helper is true, step_over_me() will spin until
        # the helper thread increments the counter.
        thread.StepOver(lldb.eOnlyThisThread)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        counter_after = target.FindFirstGlobalVariable("g_helper_count")
        val_after = counter_after.GetValueAsUnsigned()
        self.assertGreater(
            val_after,
            val_before,
            "Helper thread counter did not advance during step-over "
            "(expected it to run because of always-run-thread-names).",
        )

    @skipIfWindows
    def test_without_setting_thread_is_suspended(self):
        """Test that without the setting, the helper thread is suspended
        during single-stepping (baseline)."""
        self.build()
        (target, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Do NOT set always-run-thread-names.
        counter_before = target.FindFirstGlobalVariable("g_helper_count")
        self.assertTrue(counter_before.IsValid())
        val_before = counter_before.GetValueAsUnsigned()

        # Step over with only-this-thread mode.
        thread.StepOver(lldb.eOnlyThisThread)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        # The helper thread should have been suspended, so its counter should
        # not have changed.
        counter_after = target.FindFirstGlobalVariable("g_helper_count")
        val_after = counter_after.GetValueAsUnsigned()
        self.assertEqual(
            val_after,
            val_before,
            "Helper thread counter advanced during step-over "
            "(expected it to be suspended without always-run-thread-names).",
        )
