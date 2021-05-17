# Test the SBAPI for GetStatistics()

import json
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStatsAPI(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def test_stats_api(self):
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
        stats_json = sorted(json.loads(stream.GetData()))
        self.assertEqual(len(stats_json), 4)
        self.assertIn("Number of expr evaluation failures", stats_json)
        self.assertIn("Number of expr evaluation successes", stats_json)
        self.assertIn("Number of frame var failures", stats_json)
        self.assertIn("Number of frame var successes", stats_json)
