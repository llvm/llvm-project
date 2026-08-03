"""
Test lldb-dap terminated event
"""

import json

from lldbsuite.test.decorators import (
    skipIfRemote,
    skipIfTargetDoesNotSupportSharedLibraries,
    skipIfWindows,
)
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import InitializedEvent, LaunchArgs


@skipIfTargetDoesNotSupportSharedLibraries()
class TestDAP_eventStatistic(DAPTestCaseBase):
    """

    Test case that captures both initialized and terminated events.

    META-ONLY: Intended to succeed TestDAP_terminatedEvent.py,
    but upstream keeps updating that file, so both that and this file will probably exist for a while.
    """

    def check_statistics_summary(self, statistics):
        self.assertTrue(statistics["totalDebugInfoByteSize"] > 0)
        self.assertTrue(statistics["totalDebugInfoEnabled"] > 0)
        self.assertTrue(statistics["totalModuleCountHasDebugInfo"] > 0)

        self.assertNotIn("modules", statistics.keys())

    def check_target_summary(self, statistics):
        # lldb-dap debugs one target at a time.
        target = json.loads(statistics["targets"])[0]
        self.assertIn("totalSharedLibraryEventHitCount", target)

    @skipIfWindows
    @skipIfRemote
    def test_terminated_event(self):
        """
        Terminated Event
        Now contains the statistics of a debug session:
        metadata:
            totalDebugInfoByteSize > 0
            totalDebugInfoEnabled > 0
            totalModuleCountHasDebugInfo > 0
            ...
        """

        program_basename = "a.out.stripped"
        program = self.getBuildArtifact(program_basename)
        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program))
        session.verify_process_exited()

        terminated_event = session.wait_for_terminated_event(after=process_event)
        terminated_body = self.expect_not_none(terminated_event.body)
        statistics = terminated_body.lldb_statistics
        self.check_statistics_summary(statistics)
        self.check_target_summary(statistics)

    @skipIfWindows
    @skipIfRemote
    def test_initialized_event(self):
        """
        Initialized Event
        Now contains the statistics of a debug session:
            totalDebugInfoByteSize > 0
            totalDebugInfoEnabled > 0
            totalModuleCountHasDebugInfo > 0
            ...
        """

        program_basename = "a.out"
        program = self.getBuildArtifact(program_basename)
        session = self.build_and_create_session()
        pending_launch = session.initialize_and_launch(LaunchArgs(program))

        init_event = session.wait_for_earliest_event(InitializedEvent)
        init_body = self.expect_not_none(init_event.body)
        statistics = init_body.lldb_statistics
        self.check_statistics_summary(statistics)

        session.verify_configuration_done()
        launch_response = pending_launch.result()
        session.verify_process_exited(after=launch_response)
