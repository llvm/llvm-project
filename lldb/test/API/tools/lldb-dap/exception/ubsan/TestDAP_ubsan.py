"""
Test that we stop at runtime instrumentation locations (ubsan).
"""

from lldbsuite.test.decorators import skipUnlessUndefinedBehaviorSanitizer
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_ubsan(DAPTestCaseBase):
    @skipUnlessUndefinedBehaviorSanitizer
    def test_ubsan(self):
        """
        Test that we stop at ubsan.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program))
        stop_event = session.verify_stopped_on_exception(
            after=process_event, expected_description=r"Out of bounds index"
        )

        thread_id = self.expect_not_none(stop_event.body.threadId)
        exception_info = session.get_exception_info(thread_id)

        self.assertEqual(exception_info.breakMode, "always")
        description = self.expect_not_none(exception_info.description)
        self.assertRegex(description, r"Out of bounds index")
        self.assertEqual(exception_info.exceptionId, "runtime-instrumentation")

        # FIXME: Check on non macOS platform the stop information location heuristic
        # may be wrong. enable when we have updated Ubsan stopInfo heuristic.
        if self.platformIsDarwin():
            exception_details = self.expect_not_none(exception_info.details)
            stack_trace = self.expect_not_none(exception_details.stackTrace)
            self.assertIn("main.c", stack_trace)
