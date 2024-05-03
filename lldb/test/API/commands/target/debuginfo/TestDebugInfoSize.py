"""
Test SBTarget.GetStatistics() reporting for dwo files.
"""

import json
import os

from lldbsuite.test import lldbtest, lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test_event.build_exception import BuildError


SKELETON_DEBUGINFO_SIZE = 602
MAIN_DWO_DEBUGINFO_SIZE = 385
FOO_DWO_DEBUGINFO_SIZE = 380


class TestDebugInfoSize(lldbtest.TestBase):
    # Concurrency is the primary test factor here, not debug info variants.
    NO_DEBUG_INFO_TESTCASE = True

    def get_output_from_yaml(self):
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("a.out-main.dwo")
        foo_dwo = self.getBuildArtifact("a.out-foo.dwo")

        src_dir = self.getSourceDir()
        exe_yaml_path = os.path.join(src_dir, "a.out.yaml")
        self.yaml2obj(exe_yaml_path, exe)

        main_dwo_yaml_path = os.path.join(src_dir, "a.out-main.dwo.yaml")
        self.yaml2obj(main_dwo_yaml_path, main_dwo)

        foo_dwo_yaml_path = os.path.join(src_dir, "a.out-foo.dwo.yaml")
        self.yaml2obj(foo_dwo_yaml_path, foo_dwo)
        return (exe, main_dwo, foo_dwo)

    @add_test_categories(["dwo"])
    def test_dwo(self):
        (exe, main_dwo, foo_dwo) = self.get_output_from_yaml()

        # Make sure dwo files exist
        self.assertTrue(os.path.exists(main_dwo), f'Make sure "{main_dwo}" file exists')
        self.assertTrue(os.path.exists(foo_dwo), f'Make sure "{foo_dwo}" file exists')

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        stats = target.GetStatistics()
        stream = lldb.SBStream()
        res = stats.GetAsJSON(stream)
        debug_stats = json.loads(stream.GetData())
        self.assertIn(
            "totalDebugInfoByteSize",
            debug_stats,
            'Make sure the "totalDebugInfoByteSize" key is in target.GetStatistics()',
        )
        self.assertEqual(
            debug_stats["totalDebugInfoByteSize"],
            SKELETON_DEBUGINFO_SIZE + MAIN_DWO_DEBUGINFO_SIZE + FOO_DWO_DEBUGINFO_SIZE,
        )

    @add_test_categories(["dwo"])
    def test_only_load_skeleton_debuginfo(self):
        (exe, main_dwo, foo_dwo) = self.get_output_from_yaml()

        # REMOVE one of the dwo files
        os.unlink(main_dwo)
        os.unlink(foo_dwo)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        stats = target.GetStatistics()
        stream = lldb.SBStream()
        res = stats.GetAsJSON(stream)
        debug_stats = json.loads(stream.GetData())
        self.assertIn(
            "totalDebugInfoByteSize",
            debug_stats,
            'Make sure the "totalDebugInfoByteSize" key is in target.GetStatistics()',
        )
        self.assertEqual(debug_stats["totalDebugInfoByteSize"], SKELETON_DEBUGINFO_SIZE)

    @add_test_categories(["dwo"])
    def test_load_partial_dwos(self):
        (exe, main_dwo, foo_dwo) = self.get_output_from_yaml()

        # REMOVE one of the dwo files
        os.unlink(main_dwo)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        stats = target.GetStatistics()
        stream = lldb.SBStream()
        res = stats.GetAsJSON(stream)
        debug_stats = json.loads(stream.GetData())
        self.assertIn(
            "totalDebugInfoByteSize",
            debug_stats,
            'Make sure the "totalDebugInfoByteSize" key is in target.GetStatistics()',
        )
        self.assertEqual(
            debug_stats["totalDebugInfoByteSize"],
            SKELETON_DEBUGINFO_SIZE + FOO_DWO_DEBUGINFO_SIZE,
        )

    @add_test_categories(["dwo"])
    def test_dwos_loaded_symbols_on_demand(self):
        (exe, main_dwo, foo_dwo) = self.get_output_from_yaml()

        # Make sure dwo files exist
        self.assertTrue(os.path.exists(main_dwo), f'Make sure "{main_dwo}" file exists')
        self.assertTrue(os.path.exists(foo_dwo), f'Make sure "{foo_dwo}" file exists')

        # Load symbols on-demand
        self.runCmd("settings set symbols.load-on-demand true")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # By default dwo files will not be loaded
        stats = target.GetStatistics()
        stream = lldb.SBStream()
        res = stats.GetAsJSON(stream)
        debug_stats = json.loads(stream.GetData())
        self.assertIn(
            "totalDebugInfoByteSize",
            debug_stats,
            'Make sure the "totalDebugInfoByteSize" key is in target.GetStatistics()',
        )
        self.assertEqual(debug_stats["totalDebugInfoByteSize"], SKELETON_DEBUGINFO_SIZE)

        # Force loading all the dwo files
        stats_options = lldb.SBStatisticsOptions()
        stats_options.SetReportAllAvailableDebugInfo(True)
        stats = target.GetStatistics(stats_options)
        stream = lldb.SBStream()
        stats.GetAsJSON(stream)
        debug_stats = json.loads(stream.GetData())
        self.assertEqual(
            debug_stats["totalDebugInfoByteSize"],
            SKELETON_DEBUGINFO_SIZE + MAIN_DWO_DEBUGINFO_SIZE + FOO_DWO_DEBUGINFO_SIZE,
        )
