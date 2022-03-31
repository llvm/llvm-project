import lldb
import json
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

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

        self.expect("statistics disable", substrs=['need to enable statistics before disabling'], error=True)
        self.expect("statistics enable")
        self.expect("statistics enable", substrs=['already enabled'], error=True)
        self.expect("statistics disable")
        self.expect("statistics disable", substrs=['need to enable statistics before disabling'], error=True)

    def verify_key_in_dict(self, key, d, description):
        self.assertEqual(key in d, True,
            'make sure key "%s" is in dictionary %s' % (key, description))

    def verify_key_not_in_dict(self, key, d, description):
        self.assertEqual(key in d, False,
            'make sure key "%s" is in dictionary %s' % (key, description))

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
        self.assertEqual(success_fail_dict['successes'], num_successes,
                         'make sure success count')
        self.assertEqual(success_fail_dict['failures'], num_fails,
                         'make sure success count')

    def get_stats(self, options=None, log_path=None):
        """
            Get the output of the "statistics dump" with optional extra options
            and return the JSON as a python dictionary.
        """
        # If log_path is set, open the path and emit the output of the command
        # for debugging purposes.
        if log_path is not None:
            f = open(log_path, 'w')
        else:
            f = None
        return_obj = lldb.SBCommandReturnObject()
        command = "statistics dump "
        if options is not None:
            command += options
        if f:
            f.write('(lldb) %s\n' % (command))
        self.ci.HandleCommand(command, return_obj, False)
        metrics_json = return_obj.GetOutput()
        if f:
            f.write(metrics_json)
        return json.loads(metrics_json)


    def get_target_stats(self, debug_stats):
        if "targets" in debug_stats:
            return debug_stats["targets"][0]
        return None

    def test_expressions_frame_var_counts(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here",
                                          lldb.SBFileSpec("main.c"))

        self.expect("expr patatino", substrs=['27'])
        stats = self.get_target_stats(self.get_stats())
        self.verify_success_fail_count(stats, 'expressionEvaluation', 1, 0)
        self.expect("expr doesnt_exist", error=True,
                    substrs=["undeclared identifier 'doesnt_exist'"])
        # Doesn't successfully execute.
        self.expect("expr int *i = nullptr; *i", error=True)
        # Interpret an integer as an array with 3 elements is a failure for
        # the "expr" command, but the expression evaluation will succeed and
        # be counted as a success even though the "expr" options will for the
        # command to fail. It is more important to track expression evaluation
        # from all sources instead of just through the command, so this was
        # changed. If we want to track command success and fails, we can do
        # so using another metric.
        self.expect("expr -Z 3 -- 1", error=True,
                    substrs=["expression cannot be used with --element-count"])
        # We should have gotten 3 new failures and the previous success.
        stats = self.get_target_stats(self.get_stats())
        self.verify_success_fail_count(stats, 'expressionEvaluation', 2, 2)

        self.expect("statistics enable")
        # 'frame var' with enabled statistics will change stats.
        self.expect("frame var", substrs=['27'])
        stats = self.get_target_stats(self.get_stats())
        self.verify_success_fail_count(stats, 'frameVariable', 1, 0)

        # Test that "stopCount" is available when the process has run
        self.assertEqual('stopCount' in stats, True,
                         'ensure "stopCount" is in target JSON')
        self.assertGreater(stats['stopCount'], 0,
                           'make sure "stopCount" is greater than zero')

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
        debug_stats = self.get_stats()
        debug_stat_keys = [
            'memory',
            'modules',
            'targets',
            'totalSymbolTableParseTime',
            'totalSymbolTableIndexTime',
            'totalSymbolTablesLoadedFromCache',
            'totalSymbolTablesSavedToCache',
            'totalDebugInfoByteSize',
            'totalDebugInfoIndexTime',
            'totalDebugInfoIndexLoadedFromCache',
            'totalDebugInfoIndexSavedToCache',
            'totalDebugInfoParseTime',
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)
        stats = debug_stats['targets'][0]
        keys_exist = [
            'expressionEvaluation',
            'frameVariable',
            'moduleIdentifiers',
            'targetCreateTime',
        ]
        keys_missing = [
            'firstStopTime',
            'launchOrAttachTime'
        ]
        self.verify_keys(stats, '"stats"', keys_exist, keys_missing)
        self.assertGreater(stats['targetCreateTime'], 0.0)

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
        lldbutil.run_to_source_breakpoint(self, "// break here",
                                          lldb.SBFileSpec("main.c"))
        debug_stats = self.get_stats()
        debug_stat_keys = [
            'memory',
            'modules',
            'targets',
            'totalSymbolTableParseTime',
            'totalSymbolTableIndexTime',
            'totalSymbolTablesLoadedFromCache',
            'totalSymbolTablesSavedToCache',
            'totalDebugInfoByteSize',
            'totalDebugInfoIndexTime',
            'totalDebugInfoIndexLoadedFromCache',
            'totalDebugInfoIndexSavedToCache',
            'totalDebugInfoParseTime',
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)
        stats = debug_stats['targets'][0]
        keys_exist = [
            'expressionEvaluation',
            'firstStopTime',
            'frameVariable',
            'launchOrAttachTime',
            'moduleIdentifiers',
            'targetCreateTime',
        ]
        self.verify_keys(stats, '"stats"', keys_exist, None)
        self.assertGreater(stats['firstStopTime'], 0.0)
        self.assertGreater(stats['launchOrAttachTime'], 0.0)
        self.assertGreater(stats['targetCreateTime'], 0.0)

    def test_memory(self):
        """
            Test "statistics dump" and the memory information.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.createTestTarget(file_path=exe)
        debug_stats = self.get_stats()
        debug_stat_keys = [
            'memory',
            'modules',
            'targets',
            'totalSymbolTableParseTime',
            'totalSymbolTableIndexTime',
            'totalSymbolTablesLoadedFromCache',
            'totalSymbolTablesSavedToCache',
            'totalDebugInfoParseTime',
            'totalDebugInfoIndexTime',
            'totalDebugInfoIndexLoadedFromCache',
            'totalDebugInfoIndexSavedToCache',
            'totalDebugInfoByteSize'
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)

        memory = debug_stats['memory']
        memory_keys= [
            'strings',
        ]
        self.verify_keys(memory, '"memory"', memory_keys, None)

        strings = memory['strings']
        strings_keys= [
            'bytesTotal',
            'bytesUsed',
            'bytesUnused',
        ]
        self.verify_keys(strings, '"strings"', strings_keys, None)


    def find_module_in_metrics(self, path, stats):
        modules = stats['modules']
        for module in modules:
            if module['path'] == path:
                return module
        return None

    def find_module_by_id_in_metrics(self, id, stats):
        modules = stats['modules']
        for module in modules:
            if module['identifier'] == id:
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
            'memory',
            'modules',
            'targets',
            'totalSymbolTableParseTime',
            'totalSymbolTableIndexTime',
            'totalSymbolTablesLoadedFromCache',
            'totalSymbolTablesSavedToCache',
            'totalDebugInfoParseTime',
            'totalDebugInfoIndexTime',
            'totalDebugInfoIndexLoadedFromCache',
            'totalDebugInfoIndexSavedToCache',
            'totalDebugInfoByteSize'
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)
        stats = debug_stats['targets'][0]
        keys_exist = [
            'moduleIdentifiers',
        ]
        self.verify_keys(stats, '"stats"', keys_exist, None)
        exe_module = self.find_module_in_metrics(exe, debug_stats)
        module_keys = [
            'debugInfoByteSize',
            'debugInfoIndexLoadedFromCache',
            'debugInfoIndexTime',
            'debugInfoIndexSavedToCache',
            'debugInfoParseTime',
            'identifier',
            'path',
            'symbolTableIndexTime',
            'symbolTableLoadedFromCache',
            'symbolTableParseTime',
            'symbolTableSavedToCache',
            'triple',
            'uuid',
        ]
        self.assertNotEqual(exe_module, None)
        self.verify_keys(exe_module, 'module dict for "%s"' % (exe), module_keys)

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
            'memory',
            'modules',
            'targets',
            'totalSymbolTableParseTime',
            'totalSymbolTableIndexTime',
            'totalSymbolTablesLoadedFromCache',
            'totalSymbolTablesSavedToCache',
            'totalDebugInfoParseTime',
            'totalDebugInfoIndexTime',
            'totalDebugInfoIndexLoadedFromCache',
            'totalDebugInfoIndexSavedToCache',
            'totalDebugInfoByteSize',
        ]
        self.verify_keys(debug_stats, '"debug_stats"', debug_stat_keys, None)
        target_stats = debug_stats['targets'][0]
        keys_exist = [
            'breakpoints',
            'expressionEvaluation',
            'frameVariable',
            'targetCreateTime',
            'moduleIdentifiers',
            'totalBreakpointResolveTime',
        ]
        self.verify_keys(target_stats, '"stats"', keys_exist, None)
        self.assertGreater(target_stats['totalBreakpointResolveTime'], 0.0)
        breakpoints = target_stats['breakpoints']
        bp_keys_exist = [
            'details',
            'id',
            'internal',
            'numLocations',
            'numResolvedLocations',
            'resolveTime'
        ]
        for breakpoint in breakpoints:
            self.verify_keys(breakpoint, 'target_stats["breakpoints"]',
                             bp_keys_exist, None)


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
        exe_name = 'a.out'
        exe = self.getBuildArtifact(exe_name)
        dsym = self.getBuildArtifact(exe_name + ".dSYM")
        # Make sure the executable file exists after building.
        self.assertEqual(os.path.exists(exe), True)
        # Make sure the dSYM file exists after building.
        self.assertEqual(os.path.isdir(dsym), True)

        # Create the target
        target = self.createTestTarget(file_path=exe)

        debug_stats = self.get_stats()

        exe_stats = self.find_module_in_metrics(exe, debug_stats)
        # If we have a dSYM file, there should be a key/value pair in the module
        # statistics and the path should match the dSYM file path in the build
        # artifacts.
        self.assertIn('symbolFilePath', exe_stats)
        stats_dsym = exe_stats['symbolFilePath']

        # Make sure main executable's module info has debug info size that is
        # greater than zero as the dSYM file and main executable work together
        # in the lldb.SBModule class to provide the data.
        self.assertGreater(exe_stats['debugInfoByteSize'], 0)

        # The "dsym" variable contains the bundle directory for the dSYM, while
        # the "stats_dsym" will have the
        self.assertIn(dsym, stats_dsym)
        # Since we have a dSYM file, we should not be loading DWARF from the .o
        # files and the .o file module identifiers should NOT be in the module
        # statistics.
        self.assertNotIn('symbolFileModuleIdentifiers', exe_stats)

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
        exe_name = 'a.out'
        exe = self.getBuildArtifact(exe_name)
        dsym = self.getBuildArtifact(exe_name + ".dSYM")
        print("carp: dsym = '%s'" % (dsym))
        # Make sure the executable file exists after building.
        self.assertEqual(os.path.exists(exe), True)
        # Make sure the dSYM file doesn't exist after building.
        self.assertEqual(os.path.isdir(dsym), False)

        # Create the target
        target = self.createTestTarget(file_path=exe)

        # Force the 'main.o' .o file's DWARF to be loaded so it will show up
        # in the stats.
        self.runCmd("b main.cpp:7")

        debug_stats = self.get_stats()

        exe_stats = self.find_module_in_metrics(exe, debug_stats)
        # If we don't have a dSYM file, there should not be a key/value pair in
        # the module statistics.
        self.assertNotIn('symbolFilePath', exe_stats)

        # Make sure main executable's module info has debug info size that is
        # zero as there is no debug info in the main executable, only in the
        # .o files. The .o files will also only be loaded if something causes
        # them to be loaded, so we set a breakpoint to force the .o file debug
        # info to be loaded.
        self.assertEqual(exe_stats['debugInfoByteSize'], 0)

        # When we don't have a dSYM file, the SymbolFileDWARFDebugMap class
        # should create modules for each .o file that contains DWARF that the
        # symbol file creates, so we need to verify that we have a valid module
        # identifier for main.o that is we should not be loading DWARF from the .o
        # files and the .o file module identifiers should NOT be in the module
        # statistics.
        self.assertIn('symbolFileModuleIdentifiers', exe_stats)

        symfileIDs = exe_stats['symbolFileModuleIdentifiers']
        for symfileID in symfileIDs:
            o_module = self.find_module_by_id_in_metrics(symfileID, debug_stats)
            self.assertNotEqual(o_module, None)
            # Make sure each .o file has some debug info bytes.
            self.assertGreater(o_module['debugInfoByteSize'], 0)
