"""
Test lldb-dap threads request
"""

from lldbsuite.test.decorators import skipIfTargetDoesNotSupportThreads
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs, StoppedReason, ThreadsArgs


@skipIfTargetDoesNotSupportThreads()
class TestDAP_threads(DAPTestCaseBase):
    def test_correct_thread(self):
        """
        Tests that the correct thread is selected if we continue from
        a thread that goes away and hit a breakpoint in another thread.
        In this case, the selected thread should be the thread that
        just hit the breakpoint, and not the first thread in the list.
        """
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        source = "main.cpp"
        breakpoint_line = line_number(source, "// break here")

        with session.configure(LaunchArgs(program)) as ctx:
            breakpoint_ids = session.resolve_source_breakpoints(
                source, [breakpoint_line]
            )
        first_stop = session.verify_stopped_on_breakpoint(after=ctx.process_event)

        # We're now stopped at the breakpoint in the first thread, thread #2.
        # Continue to join the first thread and hit the breakpoint in the
        # second thread, thread #3.
        second_stop = session.continue_to_next_stop(exp_reason=StoppedReason.BREAKPOINT)
        self.assertNotEqual(
            first_stop.body.threadId,
            second_stop.body.threadId,
            "the stopped events should be on different threads.",
        )

        # Verify that the description is the relevant breakpoint,
        # preserveFocusHint is False and threadCausedFocus is True.
        stop_description = self.expect_not_none(second_stop.body.description)
        self.assertTrue(stop_description.startswith(f"breakpoint {breakpoint_ids[0]}"))
        self.assertIsNone(second_stop.body.preserveFocusHint)

        # All threads should have a name.
        threads = session.send_request(ThreadsArgs()).result().body.threads
        for t in threads:
            self.assertTrue(t.name, "thread name should be non-empty")

        session.continue_to_exit()

    def test_thread_format(self):
        """Tests the support for custom thread formats."""
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        source = "main.cpp"
        breakpoint_line = line_number(source, "// break here")

        with session.configure(
            LaunchArgs(
                program,
                customThreadFormat="This is thread index #${thread.index}",
                stopCommands=["thread list"],
            )
        ) as ctx:
            bp_ids = session.resolve_source_breakpoints(source, [breakpoint_line])
        session.verify_stopped_on_breakpoint(bp_ids, after=ctx.process_event)

        threads = session.send_request(ThreadsArgs()).result().body.threads
        if self.getPlatform() == "windows":
            # Windows creates a thread pool once WaitForSingleObject is called
            # by thread.join(). As we are in the thread function, we can't be
            # certain that join() has been called yet and a thread pool has
            # been created, thus we only check for the first two threads.
            names = sorted(t.name for t in threads)[:2]
            self.assertEqual(
                names, ["This is thread index #1", "This is thread index #2"]
            )
        else:
            self.assertEqual(threads[0].name, "This is thread index #1")
            self.assertEqual(threads[1].name, "This is thread index #2")

        # Clear the breakpoint so the second thread doesn't hit it on the way out.
        session.set_source_breakpoints(source, [])
        session.continue_to_exit()
