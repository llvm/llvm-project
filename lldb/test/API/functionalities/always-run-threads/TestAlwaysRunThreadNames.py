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

        # Record the helper thread's counter before stepping.
        counter_before = target.FindFirstGlobalVariable("g_helper_count")
        self.assertTrue(counter_before.IsValid())
        val_before = counter_before.GetValueAsUnsigned()

        # Step over -- this normally suspends all other threads.
        thread.StepOver(lldb.eOnlyThisThread)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        # The helper thread should have been allowed to run, so its counter
        # should have advanced.
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
