import glob
import json
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os
import time


class DebugIndexCacheTestcase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Set the lldb module cache directory to a directory inside the build
        # artifacts directory so no other tests are interfered with.
        self.cache_dir = os.path.join(self.getBuildDir(), 'lldb-module-cache')

    def get_module_cache_files(self, basename):
        module_cache_glob = os.path.join(self.cache_dir,
                                         "llvmcache-*%s*dwarf-index*" % (basename))
        return glob.glob(module_cache_glob)

    def get_stats(self, log_path=None):
        """
            Get the output of the "statistics dump" and return the JSON as a
            python dictionary.
        """
        # If log_path is set, open the path and emit the output of the command
        # for debugging purposes.
        if log_path is not None:
            f = open(log_path, 'w')
        else:
            f = None
        return_obj = lldb.SBCommandReturnObject()
        command = "statistics dump "
        if f:
            f.write('(lldb) %s\n' % (command))
        self.ci.HandleCommand(command, return_obj, False)
        metrics_json = return_obj.GetOutput()
        if f:
            f.write(metrics_json)
        return json.loads(metrics_json)

    def enable_lldb_index_cache(self):
        self.runCmd('settings set symbols.lldb-index-cache-path "%s"' % (self.cache_dir))
        self.runCmd('settings set symbols.enable-lldb-index-cache true')

    @no_debug_info_test
    def test_with_caching_enabled(self):
        """
            Test module cache functionality for debug info index caching.

            We test that a debug info index file is created for the debug
            information when caching is enabled with a file that contains
            at least one of each kind of DIE in ManualDWARFIndex::IndexSet.

            The input file has DWARF that will fill in every member of the
            ManualDWARFIndex::IndexSet class to ensure we can encode all of the
            required information.

            With caching enabled, we also verify that the appropriate statistics
            specify that the cache file was saved to the cache.
        """
        self.enable_lldb_index_cache()
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "exe.yaml")
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        debug_index_cache_files = self.get_module_cache_files('main.o')
        self.assertEqual(len(debug_index_cache_files), 1,
                "make sure there is one file in the module cache directory (%s) for main.o that is a debug info cache" % (self.cache_dir))

        # Verify that the module statistics have the information that specifies
        # if we loaded or saved the debug index and symtab to the cache
        stats = self.get_stats()
        module_stats = stats['modules'][0]
        self.assertFalse(module_stats['debugInfoIndexLoadedFromCache'])
        self.assertTrue(module_stats['debugInfoIndexSavedToCache'])
        self.assertFalse(module_stats['symbolTableLoadedFromCache'])
        self.assertTrue(module_stats['symbolTableSavedToCache'])
        # Verify the top level stats track how many things were loaded or saved
        # to the cache.
        self.assertEqual(stats["totalDebugInfoIndexLoadedFromCache"], 0)
        self.assertEqual(stats["totalDebugInfoIndexSavedToCache"], 1)
        self.assertEqual(stats["totalSymbolTablesLoadedFromCache"], 0)
        self.assertEqual(stats["totalSymbolTablesSavedToCache"], 1)

    @no_debug_info_test
    def test_with_caching_disabled(self):
        """
            Test module cache functionality for debug info index caching.

            We test that a debug info index file is not created for the debug
            information when caching is disabled with a file that contains
            at least one of each kind of DIE in ManualDWARFIndex::IndexSet.

            The input file has DWARF that will fill in every member of the
            ManualDWARFIndex::IndexSet class to ensure we can encode all of the
            required information.

            With caching disabled, we also verify that the appropriate
            statistics specify that the cache file was not saved to the cache.
        """
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "exe.yaml")
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        debug_index_cache_files = self.get_module_cache_files('main.o')
        self.assertEqual(len(debug_index_cache_files), 0,
                "make sure there is no file in the module cache directory (%s) for main.o that is a debug info cache" % (self.cache_dir))

        # Verify that the module statistics have the information that specifies
        # if we loaded or saved the debug index and symtab to the cache
        stats = self.get_stats()
        module_stats = stats['modules'][0]
        self.assertFalse(module_stats['debugInfoIndexLoadedFromCache'])
        self.assertFalse(module_stats['debugInfoIndexSavedToCache'])
        self.assertFalse(module_stats['symbolTableLoadedFromCache'])
        self.assertFalse(module_stats['symbolTableSavedToCache'])
        # Verify the top level stats track how many things were loaded or saved
        # to the cache.
        self.assertEqual(stats["totalDebugInfoIndexLoadedFromCache"], 0)
        self.assertEqual(stats["totalDebugInfoIndexSavedToCache"], 0)
        self.assertEqual(stats["totalSymbolTablesLoadedFromCache"], 0)
        self.assertEqual(stats["totalSymbolTablesSavedToCache"], 0)
