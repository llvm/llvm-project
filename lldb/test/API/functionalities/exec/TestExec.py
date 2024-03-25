"""
Test some lldb command abbreviations.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExecTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(
        archs=["i386"], oslist=no_match(["freebsd"]), bugnumber="rdar://28656532"
    )
    @expectedFailureAll(
        oslist=["ios", "tvos", "watchos", "bridgeos"],
        bugnumber="rdar://problem/34559552",
    )  # this exec test has problems on ios systems
    @expectedFailureNetBSD
    @skipIfAsan  # rdar://problem/43756823
    @skipIfWindows
    def test_hitting_exec(self):
        self.do_test(False)

    @expectedFailureAll(
        archs=["i386"], oslist=no_match(["freebsd"]), bugnumber="rdar://28656532"
    )
    @expectedFailureAll(
        oslist=["ios", "tvos", "watchos", "bridgeos"],
        bugnumber="rdar://problem/34559552",
    )  # this exec test has problems on ios systems
    @expectedFailureNetBSD
    @skipIfAsan  # rdar://problem/43756823
    @skipIfWindows
    def test_skipping_exec(self):
        self.do_test(True)

    def do_test(self, skip_exec):
        self.build()
        exe = self.getBuildArtifact("a.out")
        secondprog = self.getBuildArtifact("secondprog")

        # Create the target
        target = self.dbg.CreateTarget(exe)

        # Create any breakpoints we need
        breakpoint1 = target.BreakpointCreateBySourceRegex(
            "Set breakpoint 1 here", lldb.SBFileSpec("main.c", False)
        )
        self.assertTrue(breakpoint1, VALID_BREAKPOINT)
        breakpoint2 = target.BreakpointCreateBySourceRegex(
            "Set breakpoint 2 here", lldb.SBFileSpec("secondprog.cpp", False)
        )
        self.assertTrue(breakpoint2, VALID_BREAKPOINT)

        # Launch the process
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        if self.TraceOn():
            self.runCmd("settings show target.process.stop-on-exec", check=False)
        if skip_exec:
            self.dbg.HandleCommand("settings set target.process.stop-on-exec false")

            def cleanup():
                self.runCmd(
                    "settings set target.process.stop-on-exec false", check=False
                )

            # Execute the cleanup function during test case tear down.
            self.addTearDownHook(cleanup)

        # The stop reason of the thread should be breakpoint.
        self.assertState(
            process.GetState(), lldb.eStateStopped, STOPPED_DUE_TO_BREAKPOINT
        )

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint1)
        self.assertEqual(len(threads), 1)

        # We had a deadlock tearing down the TypeSystemMap on exec, but only if some
        # expression had been evaluated.  So make sure we do that here so the teardown
        # is not trivial.

        thread = threads[0]
        value = thread.frames[0].EvaluateExpression("1 + 2")
        self.assertTrue(value.IsValid(), "Expression evaluated successfully")
        int_value = value.GetValueAsSigned()
        self.assertEqual(int_value, 3, "Expression got the right result.")

        # Run and we should stop due to exec
        process.Continue()

        if not skip_exec:
            self.assertNotEqual(
                process.GetState(), lldb.eStateExited, "Process should not have exited!"
            )
            self.assertState(
                process.GetState(),
                lldb.eStateStopped,
                "Process should be stopped at __dyld_start",
            )

            threads = lldbutil.get_stopped_threads(process, lldb.eStopReasonExec)
            self.assertEqual(len(threads), 1, "We got a thread stopped for exec.")

            # Run and we should stop at breakpoint in main after exec
            process.Continue()

        self.assertState(process.GetState(), lldb.eStateStopped)
        for t in process.threads:
            if t.stop_reason != lldb.eStopReasonNone:
                self.assertStopReason(
                    t.stop_reason, lldb.eStopReasonBreakpoint, "Unexpected stop reason"
                )
                if self.TraceOn():
                    print(t)
                    if t.stop_reason != lldb.eStopReasonBreakpoint:
                        self.runCmd("bt")

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint2)
        self.assertEqual(len(threads), 1, "Stopped at breakpoint in exec'ed process.")

    @expectedFailureAll(
        archs=["i386"], oslist=no_match(["freebsd"]), bugnumber="rdar://28656532"
    )
    @expectedFailureAll(
        oslist=["ios", "tvos", "watchos", "bridgeos"],
        bugnumber="rdar://problem/34559552",
    )  # this exec test has problems on ios systems
    @expectedFailureNetBSD
    @skipIfAsan  # rdar://problem/43756823
    @skipIfWindows
    def test_correct_thread_plan_state_before_exec(self):
        """
        In this test we make sure that the Thread* cache in the ThreadPlans
        is cleared correctly when performing exec
        """

        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        (target, process, thread, breakpoint1) = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint 1 here", lldb.SBFileSpec("main.c", False)
        )

        # The stop reason of the thread should be breakpoint.
        self.assertState(
            process.GetState(), lldb.eStateStopped, STOPPED_DUE_TO_BREAKPOINT
        )

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint1)
        self.assertEqual(len(threads), 1)

        # We perform an instruction step, which effectively sets the cache of the base
        # thread plan, which should be cleared when a new thread list appears.
        #
        # Continuing after this instruction step will trigger a call to
        # ThreadPlan::ShouldReportRun, which sets the ThreadPlan's Thread cache to
        # the old Thread* value. In Process::UpdateThreadList we are clearing this
        # cache in preparation for the new ThreadList.
        #
        # Not doing this stepping will cause LLDB to first execute a private single step
        # past the current breakpoint, which eventually avoids the call to ShouldReportRun,
        # thus not setting the cache to its invalid value.
        thread.StepInstruction(False)

        # Run and we should stop due to exec
        breakpoint2 = target.BreakpointCreateBySourceRegex(
            "Set breakpoint 2 here", lldb.SBFileSpec("secondprog.cpp", False)
        )

        process.Continue()

        self.assertNotEqual(
            process.GetState(), lldb.eStateExited, "Process should not have exited!"
        )
        self.assertState(
            process.GetState(),
            lldb.eStateStopped,
            "Process should be stopped at __dyld_start",
        )

        threads = lldbutil.get_stopped_threads(process, lldb.eStopReasonExec)
        self.assertEqual(len(threads), 1, "We got a thread stopped for exec.")

        # Run and we should stop at breakpoint in main after exec
        process.Continue()

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint2)
        self.assertEqual(len(threads), 1, "Stopped at breakpoint in exec'ed process.")
