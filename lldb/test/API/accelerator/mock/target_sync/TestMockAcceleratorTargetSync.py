"""Test synchronization stops requested by an accelerator plugin."""

import lldb
from lldbsuite.test import configuration
from lldbsuite.test.decorators import add_test_categories, skipIfRemote
from lldbsuite.test.lldbtest import PROCESS_IS_VALID, VALID_TARGET, TestBase


class MockAcceleratorTargetSyncTestCase(TestBase):
    def setUp(self):
        super().setUp()
        if "mock-accelerator" not in configuration.enabled_plugins:
            self.skipTest("mock-accelerator plugin is not enabled")

    @skipIfRemote
    @add_test_categories(["llgs"])
    def test_halt_during_step_over(self):
        """An accelerator halt interrupts a native step without ending it."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        breakpoint = target.BreakpointCreateBySourceRegex(
            "// step over this line", lldb.SBFileSpec("main.c")
        )
        self.assertEqual(breakpoint.GetNumLocations(), 1)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Launch stops at the plugin's initialize breakpoint. Continuing
        # reaches the connection hook that creates the accelerator target.
        process.Continue()
        self.assertEqual(self.dbg.GetNumTargets(), 2)
        accelerator_target = next(
            candidate for candidate in self.dbg if candidate != target
        )

        process.Continue()
        thread = process.GetSelectedThread()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "main")

        thread.StepOver()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonTargetSync)
        self.assertEqual(
            thread.GetStopDescription(256),
            "target sync with target #%d"
            % self.dbg.GetIndexOfTarget(accelerator_target),
        )
        selected_target = self.dbg.GetSelectedTarget()
        self.dbg.SetSelectedTarget(target)
        self.expect(
            "process plugin packet send jThreadsInfo",
            substrs=['"reason":"target sync"', '"sync-plugin":"mock"'],
        )
        self.dbg.SetSelectedTarget(selected_target)
        functions = [frame.GetFunctionName() for frame in thread]
        self.assertIn("wait_for_accelerator_halt", functions)

        # Continuing lets the interrupted step finish.
        error = lldb.SBError()
        wait = target.FindFirstGlobalVariable("g_wait_for_accelerator_halt")
        wait.SetValueFromCString("0", error)
        self.assertSuccess(error)
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonPlanComplete)
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "main")
