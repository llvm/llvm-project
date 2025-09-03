import glob
import json
import os
import re
import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_enable_disable(self):
        """
        Test "statistics disable" and "statistics enable". These don't do
        anything anymore for cheap to gather statistics. In the future if
        statistics are expensive to gather, we can enable the feature inside
        of LLDB and test that enabling and disabling stops expesive information
        from being gathered.
        """
        self.build()
        target = self.createTestTarget()

        self.expect(
            "statistics disable",
            substrs=["need to enable statistics before disabling"],
            error=True,
        )
        self.expect("statistics enable")
        self.expect("statistics enable", substrs=["already enabled"], error=True)
        self.expect("statistics disable")
        self.expect(
            "statistics disable",
            substrs=["need to enable statistics before disabling"],
            error=True,
        )

    def verify_key_in_dict(self, key, d, description):
        self.assertIn(
            key, d, 'make sure key "%s" is in dictionary %s' % (key, description)
        )

    def verify_key_not_in_dict(self, key, d, description):
        self.assertNotIn(
            key, d, 'make sure key "%s" is in dictionary %s' % (key, description)
        )

    def verify_keys(self, dict, description, keys_exist, keys_missing=None):
        """
        Verify that all keys in "keys_exist" list are top level items in
        "dict", and that all keys in "keys_missing" do not exist as top
        level items in "dict".
        """
        if keys_exist:
            for key in keys_exist:
                self.verify_key_in_dict(key, dict, description)
        if keys_missing:
            for key in keys_missing:
                self.verify_key_not_in_dict(key, dict, description)

    def verify_success_fail_count(self, stats, key, num_successes, num_fails):
        self.verify_key_in_dict(key, stats, 'stats["%s"]' % (key))
        success_fail_dict = stats[key]
        self.assertEqual(
            success_fail_dict["successes"], num_successes, "make sure success count"
        )
        self.assertEqual(
            success_fail_dict["failures"], num_fails, "make sure success count"
        )

    def get_target_stats(self, debug_stats):
        if "targets" in debug_stats:
            return debug_stats["targets"][0]
        return None

    def get_command_stats(self, debug_stats):
        if "commands" in debug_stats:
            return debug_stats["commands"]
        return None

    def test_expressions_frame_var_counts(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect("expr patatino", substrs=["27"])
        stats = self.get_target_stats(self.get_stats())
        self.verify_success_fail_count(stats, "expressionEvaluation", 1, 0)
        self.expect(
            "expr doesnt_exist",
            error=True,
            substrs=["undeclared identifier 'doesnt_exist'"],
        )
        # Doesn't successfully execute.
        self.expect("expr int *i = nullptr; *i", error=True)
        # Interpret an integer as an array with 3 elements is a failure for
        # the "expr" command, but the expression evaluation will succeed and
        # be counted as a success even though the "expr" options will for the
        # command to fail. It is more important to track expression evaluation
        # from all sources instead of just through the command, so this was
        # changed. If we want to track command success and fails, we can do
        # so using another metric.
        self.expect(
            "expr -Z 3 -- 1",
            error=True,
            substrs=["expression cannot be used with --element-count"],
        )
        # We should have gotten 3 new failures and the previous success.
        stats = self.get_target_stats(self.get_stats())
        self.verify_success_fail_count(stats, "expressionEvaluation", 2, 2)

        self.expect("statistics enable")
        # 'frame var' with enabled statistics will change stats.
        self.expect("frame var", substrs=["27"])
        stats = self.get_target_stats(self.get_stats())
        self.verify_success_fail_count(stats, "frameVariable", 1, 0)

        # Test that "stopCount" is available when the process has run
        self.assertIn("stopCount", stats, 'ensure "stopCount" is in target JSON')
        self.assertGreater(
            stats["stopCount"], 0, 'make sure "stopCount" is greater than zero'
        )

    def test_default_no_run(self):
        """Test "statistics dump" without running the target.

        When we don't run the target, we expect to not see any 'firstStopTime'
        or 'launchOrAttachTime' top level keys that measure the launch or
        attach of the target.

        Output expected to be something like:

        (lldb) statistics dump
        {
          "memory" : {...},
          "modules" : [...],
          "targets" : [
            {
                "targetCreateTime": 0.26566899599999999,
                "expressionEvaluation": {
                    "failures": 0,
                    "successes": 0
                },
                "frameVariable": {
                    "failures": 0,
                    "successes": 0
                },
                "moduleIdentifiers": [...],
            }
          ],
          "totalDebugInfoByteSize": 182522234,
          "totalDebugInfoIndexTime": 2.33343,
          "totalDebugInfoParseTime": 8.2121400240000071,
          "totalSymbolTableParseTime": 0.123,
          "totalSymbolTableIndexTime": 0.234,
        }
        """
        self.build()
        target = self.createTestTarget()

        # Verify top-level keys.
        debug_stats = self.get_stats()
        debug_stat_keys = [
            "memory",
            "modules",
            "targets",
            "totalSymbolTableParseTime",
            "totalSymbolTableIndexTime",
            "totalSymbolTablesLoadedFromCache",
            "totalSymbolTablesSavedToCache",
            "totalSymbolTableSymbolCount",
            "totalSymbolTablesLoaded",
            "totalDebugInfoByteSize",
            "totalDebugInfoIndexTime",
            "totalDebugInfoIndexLoadedFromCache",
            "totalDebugInfoIndexSavedToCache",
            "totalDebugInfoParseTime",
            "totalDwoFileCount",
            "totalLoadedDwoFileCount",
            "totalDwoErrorCount",
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)
        if self.getPlatform() != "windows":
            self.assertGreater(debug_stats["totalSymbolTableSymbolCount"], 0)
            self.assertGreater(debug_stats["totalSymbolTablesLoaded"], 0)

        # Verify target stats keys.
        target_stats = debug_stats["targets"][0]
        target_stat_keys_exist = [
            "expressionEvaluation",
            "frameVariable",
            "moduleIdentifiers",
            "targetCreateTime",
        ]
        target_stat_keys_missing = ["firstStopTime", "launchOrAttachTime"]
        self.verify_keys(
            target_stats,
            '"target_stats"',
            target_stat_keys_exist,
            target_stat_keys_missing,
        )
        self.assertGreater(target_stats["targetCreateTime"], 0.0)

        # Verify module stats keys.
        for module_stats in debug_stats["modules"]:
            module_stat_keys_exist = [
                "symbolTableSymbolCount",
            ]
            self.verify_keys(
                module_stats, '"module_stats"', module_stat_keys_exist, None
            )
            if self.getPlatform() != "windows":
                self.assertGreater(module_stats["symbolTableSymbolCount"], 0)

    def test_default_no_run_no_preload_symbols(self):
        """Test "statistics dump" without running the target and without
        preloading symbols.

        Checks that symbol count are zero.
        """
        # Make sure symbols will not be preloaded.
        self.runCmd("settings set target.preload-symbols false")

        # Build and load the target
        self.build()
        self.createTestTarget()

        # Get statistics
        debug_stats = self.get_stats()

        # No symbols should be loaded in the main module.
        main_module_stats = debug_stats["modules"][0]
        if self.getPlatform() != "windows":
            self.assertEqual(main_module_stats["symbolTableSymbolCount"], 0)

    def test_default_with_run(self):
        """Test "statistics dump" when running the target to a breakpoint.

        When we run the target, we expect to see 'launchOrAttachTime' and
        'firstStopTime' top level keys.

        Output expected to be something like:

        (lldb) statistics dump
        {
          "memory" : {...},
          "modules" : [...],
          "targets" : [
                {
                    "firstStopTime": 0.34164492800000001,
                    "launchOrAttachTime": 0.31969605400000001,
                    "moduleIdentifiers": [...],
                    "targetCreateTime": 0.0040863039999999998
                    "expressionEvaluation": {
                        "failures": 0,
                        "successes": 0
                    },
                    "frameVariable": {
                        "failures": 0,
                        "successes": 0
                    },
                }
            ],
            "totalDebugInfoByteSize": 182522234,
            "totalDebugInfoIndexTime": 2.33343,
            "totalDebugInfoParseTime": 8.2121400240000071,
            "totalSymbolTableParseTime": 0.123,
            "totalSymbolTableIndexTime": 0.234,
        }

        """
        self.build()
        target = self.createTestTarget()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        debug_stats = self.get_stats()
        debug_stat_keys = [
            "memory",
            "modules",
            "targets",
            "totalSymbolTableParseTime",
            "totalSymbolTableIndexTime",
            "totalSymbolTablesLoadedFromCache",
            "totalSymbolTablesSavedToCache",
            "totalDebugInfoByteSize",
            "totalDebugInfoIndexTime",
            "totalDebugInfoIndexLoadedFromCache",
            "totalDebugInfoIndexSavedToCache",
            "totalDebugInfoParseTime",
            "totalDwoFileCount",
            "totalLoadedDwoFileCount",
            "totalDwoErrorCount",
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)
        stats = debug_stats["targets"][0]
        keys_exist = [
            "expressionEvaluation",
            "firstStopTime",
            "frameVariable",
            "launchOrAttachTime",
            "moduleIdentifiers",
            "targetCreateTime",
            "summaryProviderStatistics",
        ]
        self.verify_keys(stats, '"stats"', keys_exist, None)
        self.assertGreater(stats["firstStopTime"], 0.0)
        self.assertGreater(stats["launchOrAttachTime"], 0.0)
        self.assertGreater(stats["targetCreateTime"], 0.0)

    def test_memory(self):
        """
        Test "statistics dump" and the memory information.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)
        debug_stats = self.get_stats()
        debug_stat_keys = [
            "memory",
            "modules",
            "targets",
            "totalSymbolTableParseTime",
            "totalSymbolTableIndexTime",
            "totalSymbolTablesLoadedFromCache",
            "totalSymbolTablesSavedToCache",
            "totalDebugInfoParseTime",
            "totalDebugInfoIndexTime",
            "totalDebugInfoIndexLoadedFromCache",
            "totalDebugInfoIndexSavedToCache",
            "totalDebugInfoByteSize",
            "totalDwoFileCount",
            "totalLoadedDwoFileCount",
            "totalDwoErrorCount",
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)

        memory = debug_stats["memory"]
        memory_keys = [
            "strings",
        ]
        self.verify_keys(memory, '"memory"', memory_keys, None)

        strings = memory["strings"]
        strings_keys = [
            "bytesTotal",
            "bytesUsed",
            "bytesUnused",
        ]
        self.verify_keys(strings, '"strings"', strings_keys, None)

    def find_module_in_metrics(self, path, stats):
        modules = stats["modules"]
        for module in modules:
            if module["path"] == path:
                return module
        return None

    def find_module_by_id_in_metrics(self, id, stats):
        modules = stats["modules"]
        for module in modules:
            if module["identifier"] == id:
                return module
        return None

    def test_modules(self):
        """
        Test "statistics dump" and the module information.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)
        debug_stats = self.get_stats()
        debug_stat_keys = [
            "memory",
            "modules",
            "targets",
            "totalSymbolTableParseTime",
            "totalSymbolTableIndexTime",
            "totalSymbolTablesLoadedFromCache",
            "totalSymbolTablesSavedToCache",
            "totalDebugInfoParseTime",
            "totalDebugInfoIndexTime",
            "totalDebugInfoIndexLoadedFromCache",
            "totalDebugInfoIndexSavedToCache",
            "totalDebugInfoByteSize",
            "totalDwoFileCount",
            "totalLoadedDwoFileCount",
            "totalDwoErrorCount",
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)
        stats = debug_stats["targets"][0]
        keys_exist = [
            "moduleIdentifiers",
        ]
        self.verify_keys(stats, '"stats"', keys_exist, None)
        exe_module = self.find_module_in_metrics(exe, debug_stats)
        module_keys = [
            "debugInfoByteSize",
            "debugInfoIndexLoadedFromCache",
            "debugInfoIndexTime",
            "debugInfoIndexSavedToCache",
            "debugInfoParseTime",
            "identifier",
            "path",
            "symbolTableIndexTime",
            "symbolTableLoadedFromCache",
            "symbolTableParseTime",
            "symbolTableSavedToCache",
            "dwoFileCount",
            "loadedDwoFileCount",
            "dwoErrorCount",
            "triple",
            "uuid",
        ]
        self.assertNotEqual(exe_module, None)
        self.verify_keys(exe_module, 'module dict for "%s"' % (exe), module_keys)

    def test_commands(self):
        """
        Test "statistics dump" and the command information.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("target list", result)
        interp.HandleCommand("target list", result)

        debug_stats = self.get_stats()

        command_stats = self.get_command_stats(debug_stats)
        self.assertNotEqual(command_stats, None)
        self.assertEqual(command_stats["target list"], 2)

    def test_breakpoints(self):
        """Test "statistics dump"

        Output expected to be something like:

        {
          "memory" : {...},
          "modules" : [...],
          "targets" : [
                {
                    "firstStopTime": 0.34164492800000001,
                    "launchOrAttachTime": 0.31969605400000001,
                    "moduleIdentifiers": [...],
                    "targetCreateTime": 0.0040863039999999998
                    "expressionEvaluation": {
                        "failures": 0,
                        "successes": 0
                    },
                    "frameVariable": {
                        "failures": 0,
                        "successes": 0
                    },
                    "breakpoints": [
                        {
                            "details": {...},
                            "id": 1,
                            "resolveTime": 2.65438675
                        },
                        {
                            "details": {...},
                            "id": 2,
                            "resolveTime": 4.3632581669999997
                        }
                    ]
                }
            ],
            "totalDebugInfoByteSize": 182522234,
            "totalDebugInfoIndexTime": 2.33343,
            "totalDebugInfoParseTime": 8.2121400240000071,
            "totalSymbolTableParseTime": 0.123,
            "totalSymbolTableIndexTime": 0.234,
            "totalBreakpointResolveTime": 7.0176449170000001
        }

        """
        self.build()
        target = self.createTestTarget()
        self.runCmd("b main.cpp:7")
        self.runCmd("b a_function")
        debug_stats = self.get_stats()
        debug_stat_keys = [
            "memory",
            "modules",
            "targets",
            "totalSymbolTableParseTime",
            "totalSymbolTableIndexTime",
            "totalSymbolTablesLoadedFromCache",
            "totalSymbolTablesSavedToCache",
            "totalDebugInfoParseTime",
            "totalDebugInfoIndexTime",
            "totalDebugInfoIndexLoadedFromCache",
            "totalDebugInfoIndexSavedToCache",
            "totalDebugInfoByteSize",
            "totalDwoFileCount",
            "totalLoadedDwoFileCount",
            "totalDwoErrorCount",
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)
        target_stats = debug_stats["targets"][0]
        keys_exist = [
            "breakpoints",
            "expressionEvaluation",
            "frameVariable",
            "targetCreateTime",
            "moduleIdentifiers",
            "totalBreakpointResolveTime",
            "summaryProviderStatistics",
        ]
        self.verify_keys(target_stats, '"stats"', keys_exist, None)
        self.assertGreater(target_stats["totalBreakpointResolveTime"], 0.0)
        breakpoints = target_stats["breakpoints"]
        bp_keys_exist = [
            "details",
            "id",
            "internal",
            "numLocations",
            "numResolvedLocations",
            "resolveTime",
        ]
        for breakpoint in breakpoints:
            self.verify_keys(
                breakpoint, 'target_stats["breakpoints"]', bp_keys_exist, None
            )

    @add_test_categories(["dwo"])
    def test_non_split_dwarf_has_no_dwo_files(self):
        """
        Test "statistics dump" and the dwo file count.
        Builds a binary without split-dwarf mode, and then
        verifies the dwo file count is zero after running "statistics dump"
        """
        da = {"CXX_SOURCES": "third.cpp baz.cpp", "EXE": self.getBuildArtifact("a.out")}
        self.build(dictionary=da, debug_info=["debug_names"])
        self.addTearDownCleanup(dictionary=da)
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)
        debug_stats = self.get_stats()
        self.assertIn("totalDwoFileCount", debug_stats)
        self.assertIn("totalLoadedDwoFileCount", debug_stats)

        # Verify that the dwo file count is zero
        self.assertEqual(debug_stats["totalDwoFileCount"], 0)
        self.assertEqual(debug_stats["totalLoadedDwoFileCount"], 0)

    @add_test_categories(["dwo"])
    def test_no_debug_names_eager_loads_dwo_files(self):
        """
        Test the eager loading behavior of DWO files when debug_names is absent by
        building a split-dwarf binary without debug_names and then running "statistics dump".
        DWO file loading behavior:
        - With debug_names: DebugNamesDWARFIndex allows for lazy loading.
          DWO files are loaded on-demand when symbols are actually looked up
        - Without debug_names: ManualDWARFIndex uses eager loading.
          All DWO files are loaded upfront during the first symbol lookup to build a manual index.
        """
        da = {"CXX_SOURCES": "third.cpp baz.cpp", "EXE": self.getBuildArtifact("a.out")}
        self.build(dictionary=da, debug_info=["dwo"])
        self.addTearDownCleanup(dictionary=da)
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)
        debug_stats = self.get_stats()
        self.assertIn("totalDwoFileCount", debug_stats)
        self.assertIn("totalLoadedDwoFileCount", debug_stats)

        # Verify that all DWO files are loaded
        self.assertEqual(debug_stats["totalDwoFileCount"], 2)
        self.assertEqual(debug_stats["totalLoadedDwoFileCount"], 2)

    @add_test_categories(["dwo"])
    def test_split_dwarf_dwo_file_count(self):
        """
        Test "statistics dump" and the dwo file count.
        Builds a binary w/ separate .dwo files and debug_names, and then
        verifies the loaded dwo file count is the expected count after running
        various commands
        """
        da = {"CXX_SOURCES": "third.cpp baz.cpp", "EXE": self.getBuildArtifact("a.out")}
        # -gsplit-dwarf creates separate .dwo files,
        # -gpubnames enables the debug_names accelerator tables for faster symbol lookup
        #  and lazy loading of DWO files
        # Expected output: third.dwo (contains main) and baz.dwo (contains Baz struct/function)
        self.build(dictionary=da, debug_info=["dwo", "debug_names"])
        self.addTearDownCleanup(dictionary=da)
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)
        debug_stats = self.get_stats()

        # 1) 2 DWO files available but none loaded yet
        self.assertEqual(len(debug_stats["modules"]), 1)
        self.assertIn("totalLoadedDwoFileCount", debug_stats)
        self.assertIn("totalDwoFileCount", debug_stats)
        self.assertEqual(debug_stats["totalLoadedDwoFileCount"], 0)
        self.assertEqual(debug_stats["totalDwoFileCount"], 2)

        # Since there's only one module, module stats should have the same counts as total counts
        self.assertIn("dwoFileCount", debug_stats["modules"][0])
        self.assertIn("loadedDwoFileCount", debug_stats["modules"][0])
        self.assertEqual(debug_stats["modules"][0]["loadedDwoFileCount"], 0)
        self.assertEqual(debug_stats["modules"][0]["dwoFileCount"], 2)

        # 2) Setting breakpoint in main triggers loading of third.dwo (contains main function)
        self.runCmd("b main")
        debug_stats = self.get_stats()
        self.assertEqual(debug_stats["totalLoadedDwoFileCount"], 1)
        self.assertEqual(debug_stats["totalDwoFileCount"], 2)

        self.assertEqual(debug_stats["modules"][0]["loadedDwoFileCount"], 1)
        self.assertEqual(debug_stats["modules"][0]["dwoFileCount"], 2)

        # 3) Type lookup forces loading of baz.dwo (contains struct Baz definition)
        self.runCmd("type lookup Baz")
        debug_stats = self.get_stats()
        self.assertEqual(debug_stats["totalLoadedDwoFileCount"], 2)
        self.assertEqual(debug_stats["totalDwoFileCount"], 2)

        self.assertEqual(debug_stats["modules"][0]["loadedDwoFileCount"], 2)
        self.assertEqual(debug_stats["modules"][0]["dwoFileCount"], 2)

    @add_test_categories(["dwo"])
    def test_dwp_dwo_file_count(self):
        """
        Test "statistics dump" and the loaded dwo file count.
        Builds a binary w/ a separate .dwp file and debug_names, and then
        verifies the loaded dwo file count is the expected count after running
        various commands.

        We expect the DWO file counters to reflect the number of compile units
        loaded from the DWP file (each representing what was originally a separate DWO file)
        """
        da = {"CXX_SOURCES": "third.cpp baz.cpp", "EXE": self.getBuildArtifact("a.out")}
        self.build(dictionary=da, debug_info=["dwp", "debug_names"])
        self.addTearDownCleanup(dictionary=da)
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)
        debug_stats = self.get_stats()

        # Initially: 2 DWO files available but none loaded yet
        self.assertIn("totalLoadedDwoFileCount", debug_stats)
        self.assertIn("totalDwoFileCount", debug_stats)
        self.assertEqual(debug_stats["totalLoadedDwoFileCount"], 0)
        self.assertEqual(debug_stats["totalDwoFileCount"], 2)

        # Setting breakpoint in main triggers parsing of the CU within a.dwp corresponding to third.dwo (contains main function)
        self.runCmd("b main")
        debug_stats = self.get_stats()
        self.assertEqual(debug_stats["totalLoadedDwoFileCount"], 1)
        self.assertEqual(debug_stats["totalDwoFileCount"], 2)

        # Type lookup forces parsing of the CU within a.dwp corresponding to baz.dwo (contains struct Baz definition)
        self.runCmd("type lookup Baz")
        debug_stats = self.get_stats()
        self.assertEqual(debug_stats["totalDwoFileCount"], 2)
        self.assertEqual(debug_stats["totalLoadedDwoFileCount"], 2)

    @add_test_categories(["dwo"])
    def test_dwo_missing_error_stats(self):
        """
        Test that DWO missing errors are reported correctly in statistics.
        This test:
        1) Builds a program with split DWARF (.dwo files)
        2) Delete all two .dwo files
        3) Verify that 2 DWO errors are reported in statistics
        """
        da = {
            "CXX_SOURCES": "dwo_error_main.cpp dwo_error_foo.cpp",
            "EXE": self.getBuildArtifact("a.out"),
        }
        # -gsplit-dwarf creates separate .dwo files,
        # Expected output: dwo_error_main.dwo (contains main) and dwo_error_foo.dwo (contains foo struct/function)
        self.build(dictionary=da, debug_info="dwo")
        self.addTearDownCleanup(dictionary=da)
        exe = self.getBuildArtifact("a.out")

        # Remove the two .dwo files to trigger a DWO load error
        dwo_files = glob.glob(self.getBuildArtifact("*.dwo"))
        for dwo_file in dwo_files:
            os.rename(dwo_file, dwo_file + ".bak")

        target = self.createTestTarget(file_path=exe)
        debug_stats = self.get_stats()

        # Check DWO load error statistics are reported
        self.assertIn("totalDwoErrorCount", debug_stats)
        self.assertEqual(debug_stats["totalDwoErrorCount"], 2)

        # Since there's only one module, module stats should have the same count as total count
        self.assertIn("dwoErrorCount", debug_stats["modules"][0])
        self.assertEqual(debug_stats["modules"][0]["dwoErrorCount"], 2)

        # Restore the original .dwo file
        for dwo_file in dwo_files:
            os.rename(dwo_file + ".bak", dwo_file)

    @add_test_categories(["dwo"])
    def test_dwo_id_mismatch_error_stats(self):
        """
        Test that DWO ID mismatch errors are reported correctly in statistics.
        This test:
        1) Builds a program with split DWARF (.dwo files)
        2) Change one of the source file content and rebuild
        3) Replace the new .dwo file with the original one to create a DWO ID mismatch
        4) Verifies that a DWO error is reported in statistics
        5) Restores the original source file
        """
        da = {
            "CXX_SOURCES": "dwo_error_main.cpp dwo_error_foo.cpp",
            "EXE": self.getBuildArtifact("a.out"),
        }
        # -gsplit-dwarf creates separate .dwo files,
        # Expected output: dwo_error_main.dwo (contains main) and dwo_error_foo.dwo (contains foo struct/function)
        self.build(dictionary=da, debug_info="dwo")
        self.addTearDownCleanup(dictionary=da)
        exe = self.getBuildArtifact("a.out")

        # Find and make a backup of the original .dwo file
        dwo_files = glob.glob(self.getBuildArtifact("*.dwo"))

        original_dwo_file = dwo_files[1]
        original_dwo_backup = original_dwo_file + ".bak"
        shutil.copy2(original_dwo_file, original_dwo_backup)

        target = self.createTestTarget(file_path=exe)
        initial_stats = self.get_stats()
        self.assertIn("totalDwoErrorCount", initial_stats)
        self.assertEqual(initial_stats["totalDwoErrorCount"], 0)
        self.dbg.DeleteTarget(target)

        # Get the original file size before modification
        source_file_path = self.getSourcePath("dwo_error_foo.cpp")
        original_size = os.path.getsize(source_file_path)

        try:
            # Modify the source code  and rebuild
            with open(source_file_path, "a") as f:
                f.write("\n void additional_foo(){}\n")

            # Rebuild and replace the new .dwo file with the original one
            self.build(dictionary=da, debug_info="dwo")
            shutil.copy2(original_dwo_backup, original_dwo_file)

            # Create a new target and run to a breakpoint to force DWO file loading
            target = self.createTestTarget(file_path=exe)
            debug_stats = self.get_stats()

            # Check that DWO load error statistics are reported
            self.assertIn("totalDwoErrorCount", debug_stats)
            self.assertEqual(debug_stats["totalDwoErrorCount"], 1)

            # Since there's only one module, module stats should have the same count as total count
            self.assertIn("dwoErrorCount", debug_stats["modules"][0])
            self.assertEqual(debug_stats["modules"][0]["dwoErrorCount"], 1)

        finally:
            # Remove the appended content
            with open(source_file_path, "a") as f:
                f.truncate(original_size)

            # Restore the original .dwo file
            if os.path.exists(original_dwo_backup):
                os.unlink(original_dwo_backup)

    @skipUnlessDarwin
    @no_debug_info_test
    def test_dsym_binary_has_symfile_in_stats(self):
        """
        Test that if our executable has a stand alone dSYM file containing
        debug information, that the dSYM file path is listed as a key/value
        pair in the "a.out" binaries module stats. Also verify the the main
        executable's module statistics has a debug info size that is greater
        than zero as the dSYM contains debug info.
        """
        self.build(debug_info="dsym")
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)
        dsym = self.getBuildArtifact(exe_name + ".dSYM")
        # Make sure the executable file exists after building.
        self.assertTrue(os.path.exists(exe))
        # Make sure the dSYM file exists after building.
        self.assertTrue(os.path.isdir(dsym))

        # Create the target
        target = self.createTestTarget(file_path=exe)

        debug_stats = self.get_stats()

        exe_stats = self.find_module_in_metrics(exe, debug_stats)
        # If we have a dSYM file, there should be a key/value pair in the module
        # statistics and the path should match the dSYM file path in the build
        # artifacts.
        self.assertIn("symbolFilePath", exe_stats)
        stats_dsym = exe_stats["symbolFilePath"]

        # Make sure main executable's module info has debug info size that is
        # greater than zero as the dSYM file and main executable work together
        # in the lldb.SBModule class to provide the data.
        self.assertGreater(exe_stats["debugInfoByteSize"], 0)

        # The "dsym" variable contains the bundle directory for the dSYM, while
        # the "stats_dsym" will have the
        self.assertIn(dsym, stats_dsym)
        # Since we have a dSYM file, we should not be loading DWARF from the .o
        # files and the .o file module identifiers should NOT be in the module
        # statistics.
        self.assertNotIn("symbolFileModuleIdentifiers", exe_stats)

    @skipUnlessDarwin
    @no_debug_info_test
    def test_no_dsym_binary_has_symfile_identifiers_in_stats(self):
        """
        Test that if our executable loads debug info from the .o files,
        that the module statistics contains a 'symbolFileModuleIdentifiers'
        key which is a list of module identifiers, and verify that the
        module identifier can be used to find the .o file's module stats.
        Also verify the the main executable's module statistics has a debug
        info size that is zero, as the main executable itself has no debug
        info, but verify that the .o files have debug info size that is
        greater than zero. This test ensures that we don't double count
        debug info.
        """
        self.build(debug_info="dwarf")
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)
        dsym = self.getBuildArtifact(exe_name + ".dSYM")
        # Make sure the executable file exists after building.
        self.assertTrue(os.path.exists(exe))
        # Make sure the dSYM file doesn't exist after building.
        self.assertFalse(os.path.isdir(dsym))

        # Create the target
        target = self.createTestTarget(file_path=exe)

        # Force the 'main.o' .o file's DWARF to be loaded so it will show up
        # in the stats.
        self.runCmd("b main.cpp:7")

        debug_stats = self.get_stats("--all-targets")

        exe_stats = self.find_module_in_metrics(exe, debug_stats)
        # If we don't have a dSYM file, there should not be a key/value pair in
        # the module statistics.
        self.assertNotIn("symbolFilePath", exe_stats)

        # Make sure main executable's module info has debug info size that is
        # zero as there is no debug info in the main executable, only in the
        # .o files. The .o files will also only be loaded if something causes
        # them to be loaded, so we set a breakpoint to force the .o file debug
        # info to be loaded.
        self.assertEqual(exe_stats["debugInfoByteSize"], 0)

        # When we don't have a dSYM file, the SymbolFileDWARFDebugMap class
        # should create modules for each .o file that contains DWARF that the
        # symbol file creates, so we need to verify that we have a valid module
        # identifier for main.o that is we should not be loading DWARF from the .o
        # files and the .o file module identifiers should NOT be in the module
        # statistics.
        self.assertIn("symbolFileModuleIdentifiers", exe_stats)

        symfileIDs = exe_stats["symbolFileModuleIdentifiers"]
        for symfileID in symfileIDs:
            o_module = self.find_module_by_id_in_metrics(symfileID, debug_stats)
            self.assertNotEqual(o_module, None)
            # Make sure each .o file has some debug info bytes.
            self.assertGreater(o_module["debugInfoByteSize"], 0)

    @skipUnlessDarwin
    @no_debug_info_test
    def test_had_frame_variable_errors(self):
        """
        Test that if we have frame variable errors that we see this in the
        statistics for the module that had issues.
        """
        self.build(debug_info="dwarf")
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)
        dsym = self.getBuildArtifact(exe_name + ".dSYM")
        main_obj = self.getBuildArtifact("main.o")
        # Make sure the executable file exists after building.
        self.assertTrue(os.path.exists(exe))
        # Make sure the dSYM file doesn't exist after building.
        self.assertFalse(os.path.isdir(dsym))
        # Make sure the main.o object file exists after building.
        self.assertTrue(os.path.exists(main_obj))

        # Delete the main.o file that contains the debug info so we force an
        # error when we run to main and try to get variables
        os.unlink(main_obj)

        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(self, "main")

        # Get stats and verify we had errors.
        stats = self.get_stats()
        exe_stats = self.find_module_in_metrics(exe, stats)
        self.assertIsNotNone(exe_stats)

        # Make sure we have "debugInfoHadVariableErrors" variable that is set to
        # false before failing to get local variables due to missing .o file.
        self.assertFalse(exe_stats["debugInfoHadVariableErrors"])

        # Verify that the top level statistic that aggregates the number of
        # modules with debugInfoHadVariableErrors is zero
        self.assertEqual(stats["totalModuleCountWithVariableErrors"], 0)

        # Try and fail to get variables
        vars = thread.GetFrameAtIndex(0).GetVariables(True, True, False, True)

        # Make sure we got an error back that indicates that variables were not
        # available
        self.assertTrue(vars.GetError().Fail())

        # Get stats and verify we had errors.
        stats = self.get_stats()
        exe_stats = self.find_module_in_metrics(exe, stats)
        self.assertIsNotNone(exe_stats)

        # Make sure we have "hadFrameVariableErrors" variable that is set to
        # true after failing to get local variables due to missing .o file.
        self.assertTrue(exe_stats["debugInfoHadVariableErrors"])

        # Verify that the top level statistic that aggregates the number of
        # modules with debugInfoHadVariableErrors is greater than zero
        self.assertGreater(stats["totalModuleCountWithVariableErrors"], 0)

    def test_transcript_happy_path(self):
        """
        Test "statistics dump" and the transcript information.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)
        self.runCmd("settings set interpreter.save-transcript true")
        self.runCmd("version")

        # Verify the output of a first "statistics dump"
        debug_stats = self.get_stats("--transcript true")
        self.assertIn("transcript", debug_stats)
        transcript = debug_stats["transcript"]
        self.assertEqual(len(transcript), 2)
        self.assertEqual(transcript[0]["commandName"], "version")
        self.assertEqual(transcript[1]["commandName"], "statistics dump")
        # The first "statistics dump" in the transcript should have no output
        self.assertNotIn("output", transcript[1])

        # Verify the output of a second "statistics dump"
        debug_stats = self.get_stats("--transcript true")
        self.assertIn("transcript", debug_stats)
        transcript = debug_stats["transcript"]
        self.assertEqual(len(transcript), 3)
        self.assertEqual(transcript[0]["commandName"], "version")
        self.assertEqual(transcript[1]["commandName"], "statistics dump")
        # The first "statistics dump" in the transcript should have output now
        self.assertIn("output", transcript[1])
        self.assertEqual(transcript[2]["commandName"], "statistics dump")
        # The second "statistics dump" in the transcript should have no output
        self.assertNotIn("output", transcript[2])

    def test_transcript_warning_when_disabled(self):
        """
        Test that "statistics dump --transcript=true" shows a warning when
        transcript saving is disabled.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)

        # Ensure transcript saving is disabled (this is the default)
        self.runCmd("settings set interpreter.save-transcript false")

        # Request transcript in statistics dump and check for warning
        interpreter = self.dbg.GetCommandInterpreter()
        res = lldb.SBCommandReturnObject()
        interpreter.HandleCommand("statistics dump --transcript=true", res)
        self.assertTrue(res.Succeeded())
        # We should warn about transcript being requested but not saved
        self.assertIn(
            "transcript requested but none was saved. Enable with "
            "'settings set interpreter.save-transcript true'",
            res.GetError(),
        )

    def verify_stats(self, stats, expectation, options):
        for field_name in expectation:
            idx = field_name.find(".")
            if idx == -1:
                # `field_name` is a top-level field
                exists = field_name in stats
                should_exist = expectation[field_name]
                should_exist_string = "" if should_exist else "not "
                self.assertEqual(
                    exists,
                    should_exist,
                    f"'{field_name}' should {should_exist_string}exist for 'statistics dump{options}'",
                )
            else:
                # `field_name` is a string of "<top-level field>.<second-level field>"
                top_level_field_name = field_name[0:idx]
                second_level_field_name = field_name[idx + 1 :]
                for top_level_field in (
                    stats[top_level_field_name] if top_level_field_name in stats else {}
                ):
                    exists = second_level_field_name in top_level_field
                    should_exist = expectation[field_name]
                    should_exist_string = "" if should_exist else "not "
                    self.assertEqual(
                        exists,
                        should_exist,
                        f"'{field_name}' should {should_exist_string}exist for 'statistics dump{options}'",
                    )

    def get_test_cases_for_sections_existence(self):
        should_always_exist_or_not = {
            "totalDebugInfoEnabled": True,
            "memory": True,
        }
        test_cases = [
            {  # Everything mode
                "command_options": "",
                "api_options": {},
                "expect": {
                    "commands": True,
                    "targets": True,
                    "targets.moduleIdentifiers": True,
                    "targets.breakpoints": True,
                    "targets.expressionEvaluation": True,
                    "targets.frameVariable": True,
                    "targets.totalSharedLibraryEventHitCount": True,
                    "modules": True,
                    "transcript": False,
                },
            },
            {  # Summary mode
                "command_options": " --summary",
                "api_options": {
                    "SetSummaryOnly": True,
                },
                "expect": {
                    "commands": False,
                    "targets": True,
                    "targets.moduleIdentifiers": False,
                    "targets.breakpoints": False,
                    "targets.expressionEvaluation": False,
                    "targets.frameVariable": False,
                    "targets.totalSharedLibraryEventHitCount": True,
                    "modules": False,
                    "transcript": False,
                },
            },
            {  # Summary mode with targets
                "command_options": " --summary --targets=true",
                "api_options": {
                    "SetSummaryOnly": True,
                    "SetIncludeTargets": True,
                },
                "expect": {
                    "commands": False,
                    "targets": True,
                    "targets.moduleIdentifiers": False,
                    "targets.breakpoints": False,
                    "targets.expressionEvaluation": False,
                    "targets.frameVariable": False,
                    "targets.totalSharedLibraryEventHitCount": True,
                    "modules": False,
                    "transcript": False,
                },
            },
            {  # Summary mode without targets
                "command_options": " --summary --targets=false",
                "api_options": {
                    "SetSummaryOnly": True,
                    "SetIncludeTargets": False,
                },
                "expect": {
                    "commands": False,
                    "targets": False,
                    "modules": False,
                    "transcript": False,
                },
            },
            {  # Summary mode with modules
                "command_options": " --summary --modules=true",
                "api_options": {
                    "SetSummaryOnly": True,
                    "SetIncludeModules": True,
                },
                "expect": {
                    "commands": False,
                    "targets": True,
                    "targets.moduleIdentifiers": False,
                    "targets.breakpoints": False,
                    "targets.expressionEvaluation": False,
                    "targets.frameVariable": False,
                    "targets.totalSharedLibraryEventHitCount": True,
                    "modules": True,
                    "transcript": False,
                },
            },
            {  # Default mode without modules and transcript
                "command_options": " --modules=false --transcript=false",
                "api_options": {
                    "SetIncludeModules": False,
                    "SetIncludeTranscript": False,
                },
                "expect": {
                    "commands": True,
                    "targets": True,
                    "targets.moduleIdentifiers": False,
                    "targets.breakpoints": True,
                    "targets.expressionEvaluation": True,
                    "targets.frameVariable": True,
                    "targets.totalSharedLibraryEventHitCount": True,
                    "modules": False,
                    "transcript": False,
                },
            },
            {  # Default mode without modules and with transcript
                "command_options": " --modules=false --transcript=true",
                "api_options": {
                    "SetIncludeModules": False,
                    "SetIncludeTranscript": True,
                },
                "expect": {
                    "commands": True,
                    "targets": True,
                    "targets.moduleIdentifiers": False,
                    "targets.breakpoints": True,
                    "targets.expressionEvaluation": True,
                    "targets.frameVariable": True,
                    "targets.totalSharedLibraryEventHitCount": True,
                    "modules": False,
                    "transcript": True,
                },
            },
            {  # Default mode without modules
                "command_options": " --modules=false",
                "api_options": {
                    "SetIncludeModules": False,
                },
                "expect": {
                    "commands": True,
                    "targets": True,
                    "targets.moduleIdentifiers": False,
                    "targets.breakpoints": True,
                    "targets.expressionEvaluation": True,
                    "targets.frameVariable": True,
                    "targets.totalSharedLibraryEventHitCount": True,
                    "modules": False,
                    "transcript": False,
                },
            },
            {  # Default mode with transcript
                "command_options": " --transcript=true",
                "api_options": {
                    "SetIncludeTranscript": True,
                },
                "expect": {
                    "commands": True,
                    "targets": True,
                    "targets.moduleIdentifiers": True,
                    "targets.breakpoints": True,
                    "targets.expressionEvaluation": True,
                    "targets.frameVariable": True,
                    "targets.totalSharedLibraryEventHitCount": True,
                    "modules": True,
                    "transcript": True,
                },
            },
        ]
        return (should_always_exist_or_not, test_cases)

    def test_sections_existence_through_command(self):
        """
        Test "statistics dump" and the existence of sections when different
        options are given through the command line (CLI or HandleCommand).
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)

        # Create some transcript so that it can be tested.
        self.runCmd("settings set interpreter.save-transcript true")
        self.runCmd("version")
        self.runCmd("b main")
        # Then disable transcript so that it won't change during verification
        self.runCmd("settings set interpreter.save-transcript false")

        # Expectation
        (
            should_always_exist_or_not,
            test_cases,
        ) = self.get_test_cases_for_sections_existence()

        # Verification
        for test_case in test_cases:
            options = test_case["command_options"]
            # Get statistics dump result
            stats = self.get_stats(options)
            # Verify that each field should exist (or not)
            expectation = {**should_always_exist_or_not, **test_case["expect"]}
            self.verify_stats(stats, expectation, options)

    def test_sections_existence_through_api(self):
        """
        Test "statistics dump" and the existence of sections when different
        options are given through the public API.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)

        # Create some transcript so that it can be tested.
        self.runCmd("settings set interpreter.save-transcript true")
        self.runCmd("version")
        self.runCmd("b main")
        # But disable transcript so that it won't change during verification
        self.runCmd("settings set interpreter.save-transcript false")

        # Expectation
        (
            should_always_exist_or_not,
            test_cases,
        ) = self.get_test_cases_for_sections_existence()

        # Verification
        for test_case in test_cases:
            # Create options
            options = test_case["api_options"]
            sb_options = lldb.SBStatisticsOptions()
            for method_name, param_value in options.items():
                getattr(sb_options, method_name)(param_value)
            # Get statistics dump result
            stream = lldb.SBStream()
            target.GetStatistics(sb_options).GetAsJSON(stream)
            stats = json.loads(stream.GetData())
            # Verify that each field should exist (or not)
            expectation = {**should_always_exist_or_not, **test_case["expect"]}
            self.verify_stats(stats, expectation, options)

    def test_order_of_options_do_not_matter(self):
        """
        Test "statistics dump" and the order of options.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)

        # Create some transcript so that it can be tested.
        self.runCmd("settings set interpreter.save-transcript true")
        self.runCmd("version")
        self.runCmd("b main")
        # Then disable transcript so that it won't change during verification
        self.runCmd("settings set interpreter.save-transcript false")

        # The order of the following options shouldn't matter
        test_cases = [
            (" --summary", " --targets=true"),
            (" --summary", " --targets=false"),
            (" --summary", " --modules=true"),
            (" --summary", " --modules=false"),
            (" --summary", " --transcript=true"),
            (" --summary", " --transcript=false"),
        ]

        # Verification
        for options in test_cases:
            debug_stats_0 = self.get_stats(options[0] + options[1])
            debug_stats_1 = self.get_stats(options[1] + options[0])
            # Redact all numbers
            debug_stats_0 = re.sub(r"\d+", "0", json.dumps(debug_stats_0))
            debug_stats_1 = re.sub(r"\d+", "0", json.dumps(debug_stats_1))
            # Verify that the two output are the same
            self.assertEqual(
                debug_stats_0,
                debug_stats_1,
                f"The order of options '{options[0]}' and '{options[1]}' should not matter",
            )

    @skipIfWindows
    def test_summary_statistics_providers(self):
        """
        Test summary timing statistics is included in statistics dump when
        a type with a summary provider exists, and is evaluated.
        """

        self.build()
        target = self.createTestTarget()
        lldbutil.run_to_source_breakpoint(
            self, "// stop here", lldb.SBFileSpec("main.cpp")
        )
        self.expect("frame var", substrs=["hello world"])
        stats = self.get_target_stats(self.get_stats())
        self.assertIn("summaryProviderStatistics", stats)
        summary_providers = stats["summaryProviderStatistics"]
        # We don't want to take a dependency on the type name, so we just look
        # for string and that it was called once.
        summary_provider_str = str(summary_providers)
        self.assertIn("string", summary_provider_str)
        self.assertIn("'count': 1", summary_provider_str)
        self.assertIn("'totalTime':", summary_provider_str)
        # We may hit the std::string C++ provider, or a summary provider string
        self.assertIn("'type':", summary_provider_str)
        self.assertTrue(
            "c++" in summary_provider_str or "string" in summary_provider_str
        )

        self.runCmd("continue")
        self.runCmd("command script import BoxFormatter.py")
        self.expect("frame var", substrs=["box = [27]"])
        stats = self.get_target_stats(self.get_stats())
        self.assertIn("summaryProviderStatistics", stats)
        summary_providers = stats["summaryProviderStatistics"]
        summary_provider_str = str(summary_providers)
        self.assertIn("BoxFormatter.summary", summary_provider_str)
        self.assertIn("'count': 1", summary_provider_str)
        self.assertIn("'totalTime':", summary_provider_str)
        self.assertIn("'type': 'python'", summary_provider_str)

    @skipIfWindows
    def test_summary_statistics_providers_vec(self):
        """
        Test summary timing statistics is included in statistics dump when
        a type with a summary provider exists, and is evaluated. This variation
        tests that vector recurses into it's child type.
        """
        self.build()
        target = self.createTestTarget()
        lldbutil.run_to_source_breakpoint(
            self, "// stop vector", lldb.SBFileSpec("main.cpp")
        )
        self.expect(
            "frame var", substrs=["int_vec", "double_vec", "[0] = 1", "[7] = 8"]
        )
        stats = self.get_target_stats(self.get_stats())
        self.assertIn("summaryProviderStatistics", stats)
        summary_providers = stats["summaryProviderStatistics"]
        summary_provider_str = str(summary_providers)
        self.assertIn("'count': 2", summary_provider_str)
        self.assertIn("'totalTime':", summary_provider_str)
        self.assertIn("'type':", summary_provider_str)
        # We may hit the std::vector C++ provider, or a summary provider string
        if "c++" in summary_provider_str:
            self.assertIn("std::vector", summary_provider_str)

    @skipIfWindows
    def test_multiple_targets(self):
        """
        Test statistics dump only reports the stats from current target and
        "statistics dump --all-targets" includes all target stats.
        """
        da = {"CXX_SOURCES": "main.cpp", "EXE": self.getBuildArtifact("a.out")}
        self.build(dictionary=da)
        self.addTearDownCleanup(dictionary=da)

        db = {"CXX_SOURCES": "second.cpp", "EXE": self.getBuildArtifact("second.out")}
        self.build(dictionary=db)
        self.addTearDownCleanup(dictionary=db)

        main_exe = self.getBuildArtifact("a.out")
        second_exe = self.getBuildArtifact("second.out")

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp"), None, "a.out"
        )
        debugger_stats1 = self.get_stats()
        self.assertIsNotNone(self.find_module_in_metrics(main_exe, debugger_stats1))
        self.assertIsNone(self.find_module_in_metrics(second_exe, debugger_stats1))

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("second.cpp"), None, "second.out"
        )
        debugger_stats2 = self.get_stats()
        self.assertIsNone(self.find_module_in_metrics(main_exe, debugger_stats2))
        self.assertIsNotNone(self.find_module_in_metrics(second_exe, debugger_stats2))

        all_targets_stats = self.get_stats("--all-targets")
        self.assertIsNotNone(self.find_module_in_metrics(main_exe, all_targets_stats))
        self.assertIsNotNone(self.find_module_in_metrics(second_exe, all_targets_stats))

    # Return some level of the plugin stats hierarchy.
    # Will return either the top-level node, the namespace node, or a specific
    # plugin node based on requested values.
    #
    # If any of the requested keys are not found in the stats then return None.
    #
    # Plugin stats look like this:
    #
    # "plugins": {
    #     "system-runtime": [
    #         {
    #             "enabled": true,
    #             "name": "systemruntime-macosx"
    #         }
    #     ]
    # },
    def get_plugin_stats(self, debugger_stats, plugin_namespace=None, plugin_name=None):
        # Get top level plugin stats.
        if "plugins" not in debugger_stats:
            return None
        plugins = debugger_stats["plugins"]
        if not plugin_namespace:
            return plugins

        # Plugin namespace stats.
        if plugin_namespace not in plugins:
            return None
        plugins_for_namespace = plugins[plugin_namespace]
        if not plugin_name:
            return plugins_for_namespace

        # Specific plugin stats.
        for plugin in debugger_stats["plugins"][plugin_namespace]:
            if plugin["name"] == plugin_name:
                return plugin
        return None

    def test_plugin_stats(self):
        """
        Test "statistics dump" contains plugin info.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)
        debugger_stats = self.get_stats()

        # Verify that the statistics dump contains the plugin information.
        plugins = self.get_plugin_stats(debugger_stats)
        self.assertIsNotNone(plugins)

        # Check for a known plugin namespace that should be in the stats.
        system_runtime_plugins = self.get_plugin_stats(debugger_stats, "system-runtime")
        self.assertIsNotNone(system_runtime_plugins)

        # Validate the keys exists for the bottom-level plugin stats.
        plugin_keys_exist = [
            "name",
            "enabled",
        ]
        for plugin in system_runtime_plugins:
            self.verify_keys(
                plugin, 'debugger_stats["plugins"]["system-runtime"]', plugin_keys_exist
            )

        # Check for a known plugin that is enabled by default.
        system_runtime_macosx_plugin = self.get_plugin_stats(
            debugger_stats, "system-runtime", "systemruntime-macosx"
        )
        self.assertIsNotNone(system_runtime_macosx_plugin)
        self.assertTrue(system_runtime_macosx_plugin["enabled"])

        # Now disable the plugin and check the stats again.
        # The stats should show the plugin is disabled.
        self.runCmd("plugin disable system-runtime.systemruntime-macosx")
        debugger_stats = self.get_stats()
        system_runtime_macosx_plugin = self.get_plugin_stats(
            debugger_stats, "system-runtime", "systemruntime-macosx"
        )
        self.assertIsNotNone(system_runtime_macosx_plugin)
        self.assertFalse(system_runtime_macosx_plugin["enabled"])

        # Plugins should not show up in the stats when disabled with an option.
        debugger_stats = self.get_stats("--plugins false")
        plugins = self.get_plugin_stats(debugger_stats)
        self.assertIsNone(plugins)
