"""
Test that a thread created while lldb runs only one thread stays suspended
until lldb lets it run.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireThreadSupport
@requireWindows
class StepOnlyThisThreadTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def start(self, single_thread_plan_timeout):
        # After this many milliseconds a step that runs only one thread lets
        # all threads run, to break deadlocks. It defaults to 1000.
        self.runCmd(
            "settings set target.process.thread.single-thread-plan-timeout %d"
            % single_thread_plan_timeout
        )
        self.addTearDownHook(
            lambda: self.runCmd(
                "settings clear target.process.thread.single-thread-plan-timeout"
            )
        )

        self.build()
        self.source = lldb.SBFileSpec("main.cpp")
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// Start the worker.", self.source
        )
        worker_bp = target.BreakpointCreateBySourceRegex(
            "// Break in the worker.", self.source
        )
        self.assertEqual(worker_bp.GetNumLocations(), 1)
        return target, process, thread, worker_bp

    def test_step_over_keeps_new_thread_suspended(self):
        target, process, thread, worker_bp = self.start(single_thread_plan_timeout=0)

        # start_worker() creates the worker, then sleeps for a second. If the
        # worker ran during the step, it would hit its breakpoint and stop the
        # process in the middle of the step.
        thread.StepOver(lldb.eOnlyThisThread)
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonPlanComplete)
        self.assertEqual(
            thread.frames[0].GetLineEntry().GetLine(),
            line_number("main.cpp", "// Stepped to here."),
        )
        self.assertEqual(worker_bp.GetHitCount(), 0)
        worker_ran = target.FindFirstGlobalVariable("g_worker_ran")
        self.assertTrue(worker_ran.IsValid())
        self.assertEqual(worker_ran.GetValueAsUnsigned(), 0)

        # The worker was only held back: it runs once all threads do.
        threads = lldbutil.continue_to_breakpoint(process, worker_bp)
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0].GetThreadID(), thread.GetThreadID())

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)

    @skipIfWindowsAndNoLLDBServer
    def test_single_thread_timeout_releases_new_thread(self):
        target, process, thread, worker_bp = self.start(single_thread_plan_timeout=100)

        thread.StepOver(lldb.eOnlyThisThread)
        self.assertState(process.GetState(), lldb.eStateStopped)
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, worker_bp)
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0].GetThreadID(), thread.GetThreadID())

    def test_expression_waits_for_new_thread(self):
        self.start(single_thread_plan_timeout=0)
        self.expect_expr("run_and_join_thread()", result_type="int", result_value="42")

    def test_new_thread_runs_while_another_is_suspended(self):
        target, process, thread, worker_bp = self.start(single_thread_plan_timeout=0)

        bystanders = [
            t
            for t in lldbutil.get_threads_in_executable(process)
            if t.GetThreadID() != thread.GetThreadID()
        ]
        self.assertEqual(len(bystanders), 1)
        bystander = bystanders[0]
        bystander.Suspend()

        threads = lldbutil.continue_to_breakpoint(process, worker_bp)
        self.assertEqual(len(threads), 1)
        self.assertNotIn(
            threads[0].GetThreadID(), [thread.GetThreadID(), bystander.GetThreadID()]
        )

        bystander.Resume()
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
