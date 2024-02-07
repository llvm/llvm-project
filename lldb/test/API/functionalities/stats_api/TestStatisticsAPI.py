# Test the SBAPI for GetStatistics()

import json
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStatsAPI(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_stats_api(self):
        """
        Test SBTarget::GetStatistics() API.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        # Test enabling/disabling stats
        self.assertFalse(target.GetCollectingStats())
        target.SetCollectingStats(True)
        self.assertTrue(target.GetCollectingStats())
        target.SetCollectingStats(False)
        self.assertFalse(target.GetCollectingStats())

        # Test the function to get the statistics in JSON'ish.
        stats = target.GetStatistics()
        stream = lldb.SBStream()
        res = stats.GetAsJSON(stream)
        debug_stats = json.loads(stream.GetData())
        self.assertEqual(
            "targets" in debug_stats,
            True,
            'Make sure the "targets" key in in target.GetStatistics()',
        )
        self.assertEqual(
            "modules" in debug_stats,
            True,
            'Make sure the "modules" key in in target.GetStatistics()',
        )
        stats_json = debug_stats["targets"][0]
        self.assertEqual(
            "expressionEvaluation" in stats_json,
            True,
            'Make sure the "expressionEvaluation" key in in target.GetStatistics()["targets"][0]',
        )
        self.assertEqual(
            "frameVariable" in stats_json,
            True,
            'Make sure the "frameVariable" key in in target.GetStatistics()["targets"][0]',
        )
        expressionEvaluation = stats_json["expressionEvaluation"]
        self.assertEqual(
            "successes" in expressionEvaluation,
            True,
            'Make sure the "successes" key in in "expressionEvaluation" dictionary"',
        )
        self.assertEqual(
            "failures" in expressionEvaluation,
            True,
            'Make sure the "failures" key in in "expressionEvaluation" dictionary"',
        )
        frameVariable = stats_json["frameVariable"]
        self.assertEqual(
            "successes" in frameVariable,
            True,
            'Make sure the "successes" key in in "frameVariable" dictionary"',
        )
        self.assertEqual(
            "failures" in frameVariable,
            True,
            'Make sure the "failures" key in in "frameVariable" dictionary"',
        )

    def test_command_stats_api(self):
        """
        Test GetCommandInterpreter::GetStatistics() API.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        lldbutil.run_to_name_breakpoint(self, "main")

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("bt", result)

        stream = lldb.SBStream()
        res = interp.GetStatistics().GetAsJSON(stream)
        command_stats = json.loads(stream.GetData())

        # Verify bt command is correctly parsed into final form.
        self.assertEqual(command_stats["thread backtrace"], 1)
        # Verify original raw command is not duplicatedly captured.
        self.assertNotIn("bt", command_stats)
        # Verify bt's regex command is not duplicatedly captured.
        self.assertNotIn("_regexp-bt", command_stats)
