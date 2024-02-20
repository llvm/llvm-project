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
        # Launch a process and break
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

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

        # Test statistics summary.
        stats_options = lldb.SBStatisticsOptions()
        stats_options.SetSummaryOnly(True)
        stats_summary = target.GetStatistics(stats_options)
        stream_summary = lldb.SBStream()
        stats_summary.GetAsJSON(stream_summary)
        debug_stats_summary = json.loads(stream_summary.GetData())
        self.assertNotIn("modules", debug_stats_summary)
        self.assertNotIn("memory", debug_stats_summary)
        self.assertNotIn("commands", debug_stats_summary)

        # Summary values should be the same as in full statistics.
        # Except the parse time on Mac OS X is not deterministic.
        for key, value in debug_stats_summary.items():
            self.assertIn(key, debug_stats)
            if key != "targets" and not key.endswith("Time"):
                self.assertEqual(debug_stats[key], value)

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

    @add_test_categories(["dwo"])
    def test_command_stats_force(self):
        """
        Test reporting all pssible debug info stats by force loading all debug
        info. For example, dwo files
        """
        src_dir = self.getSourceDir()
        dwo_yaml_path = os.path.join(src_dir, "main-main.dwo.yaml")
        exe_yaml_path = os.path.join(src_dir, "main.yaml")
        dwo_path = self.getBuildArtifact("main-main.dwo")
        exe_path = self.getBuildArtifact("main")
        self.yaml2obj(dwo_yaml_path, dwo_path)
        self.yaml2obj(exe_yaml_path, exe_path)

        # Turn on symbols on-demand loading
        self.runCmd("settings set symbols.load-on-demand true")

        # We need the current working directory to be set to the build directory
        os.chdir(self.getBuildDir())
        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target, VALID_TARGET)

        # Get statistics
        stats_options = lldb.SBStatisticsOptions()
        stats = target.GetStatistics(stats_options)
        stream = lldb.SBStream()
        stats.GetAsJSON(stream)
        debug_stats = json.loads(stream.GetData())
        self.assertEqual(debug_stats["totalDebugInfoByteSize"], 193)

        # Get statistics with force loading
        stats_options.SetReportAllAvailableDebugInfo(True)
        stats_force = target.GetStatistics(stats_options)
        stream_force = lldb.SBStream()
        stats_force.GetAsJSON(stream_force)
        debug_stats_force = json.loads(stream_force.GetData())
        self.assertEqual(debug_stats_force["totalDebugInfoByteSize"], 445)
