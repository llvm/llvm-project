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


class TestDAP_terminatedEvent(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_initialized_event(self):
        """
        Initialized Event
        Now contains the statistics of a debug session:
        memory:
            strings
                bytesTotal > 0
            ...
        targets:
            list
        totalSymbolTableParseTime int:
        totalSymbolTablesLoadedFromCache int:
        """

        program_basename = "a.out.stripped"
        program = self.getBuildArtifact(program_basename)
        self.build_and_launch(program)

        self.continue_to_next_stop()

        initialized_event = next(
            (x for x in self.dap_server.startup_events if x["event"] == "initialized"),
            None,
        )
        self.assertIsNotNone(initialized_event)

        statistics = initialized_event["body"]["$__lldb_statistics"]
        self.assertGreater(statistics["memory"]["strings"]["bytesTotal"], 0)

        self.assertIn("targets", statistics.keys())
        self.assertIn("totalSymbolTableParseTime", statistics.keys())
