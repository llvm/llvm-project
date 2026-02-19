"""
Test lldb-dap terminated event
"""

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import json
import re

import lldbdap_testcase
from lldbsuite.test import lldbutil


class TestDAP_eventStatistic(lldbdap_testcase.DAPTestCaseBase):
    """

    Test case that captures both initialized and terminated events.

    META-ONLY: Intended to succeed TestDAP_terminatedEvent.py, but upstream keeps updating that file, so both that and this file will probably exist for a while.

    """

    def check_statistics_summary(self, statistics):
        self.assertTrue(statistics["totalDebugInfoByteSize"] > 0)
        self.assertTrue(statistics["totalDebugInfoEnabled"] > 0)
        self.assertTrue(statistics["totalModuleCountHasDebugInfo"] > 0)

        self.assertNotIn("modules", statistics.keys())

    def check_target_summary(self, statistics):
        # lldb-dap debugs one target at a time
        target = json.loads(statistics["targets"])[0]
        self.assertIn("totalSharedLibraryEventHitCount", target)

    @skipIfWindows
    @skipIfRemote
    def test_terminated_event(self):
        """
        Terminated Event
        Now contains the statistics of a debug session:
        metatdata:
            totalDebugInfoByteSize > 0
            totalDebugInfoEnabled > 0
            totalModuleCountHasDebugInfo > 0
            ...
        """

        program_basename = "a.out.stripped"
        program = self.getBuildArtifact(program_basename)
        self.build_and_launch(program)
        self.continue_to_exit()

        statistics = self.dap_server.wait_for_terminated()["body"]["$__lldb_statistics"]
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
        self.build_and_launch(program)
        self.dap_server.wait_for_event("initialized")
        statistics = self.dap_server.initialized_event["body"]["$__lldb_statistics"]
        self.check_statistics_summary(statistics)
        self.continue_to_exit()
