"""
Test exception behavior in DAP with signal.
"""

from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.decorators import skipIfNoSignals
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


@skipIfNoSignals
class TestDAP_exception(DAPTestCaseBase):
    def test_stopped_description(self):
        """
        Test that exception description is shown correctly in stopped
        event.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program=program))

        stopped_event = session.verify_stopped_on_exception(
            expected_description="signal SIGABRT", after=process_event
        )
        thread_id = self.expect_not_none(stopped_event.body.threadId)
        exception_info = session.get_exception_info(thread_id)

        self.assertEqual(exception_info.breakMode, "always")
        description = self.expect_not_none(exception_info.description)
        self.assertEqual(description, "signal SIGABRT")
        self.assertEqual(exception_info.exceptionId, "signal")
