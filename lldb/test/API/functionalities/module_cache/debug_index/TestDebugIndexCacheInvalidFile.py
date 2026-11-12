"""Test that a debug info index cache file that can't be decoded is replaced."""

import glob
import json
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os


class DebugIndexCacheInvalidFileTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False

    def get_module_cache_files(self, basename):
        module_cache_glob = os.path.join(
            self.cache_dir, "llvmcache-*%s*dwarf-index*" % (basename)
        )
        return glob.glob(module_cache_glob)

    def get_debug_info_index_stats(self, obj_path):
        """
        Return the main.o module statistics from a newly created target, after
        releasing the modules of any previous target.
        """
        lldb.SBModule.GarbageCollectAllocatedModules()
        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)
        return_obj = lldb.SBCommandReturnObject()
        self.ci.HandleCommand("statistics dump", return_obj, False)
        self.assertTrue(return_obj.Succeeded())
        self.dbg.DeleteTarget(target)
        stats = json.loads(return_obj.GetOutput())
        for module_stats in stats["modules"]:
            if module_stats["path"].endswith("main.o"):
                return module_stats
        self.fail("no module statistics for main.o")

    @no_debug_info_test
    def test(self):
        """
        Test that a debug info index cache file that fails to decode is
        removed so that a new one can be saved in its place.
        """
        # The index cache path is latched on first use, so this must be the
        # only test in this file.
        self.cache_dir = os.path.join(self.getBuildDir(), "lldb-module-cache")
        self.runCmd(
            'settings set symbols.lldb-index-cache-path "%s"' % (self.cache_dir)
        )
        self.runCmd("settings set symbols.enable-lldb-index-cache true")
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(os.path.join(self.getSourceDir(), "exe.yaml"), obj_path)

        module_stats = self.get_debug_info_index_stats(obj_path)
        self.assertTrue(module_stats["debugInfoIndexSavedToCache"])
        cache_files = self.get_module_cache_files("main.o")
        self.assertEqual(len(cache_files), 1)

        # SaveToCache doesn't overwrite an existing cache file, so the
        # truncated file must be removed for a new one to be saved.
        with open(cache_files[0], "r+b") as f:
            f.truncate(os.path.getsize(cache_files[0]) // 2)
        module_stats = self.get_debug_info_index_stats(obj_path)
        self.assertFalse(module_stats["debugInfoIndexLoadedFromCache"])
        self.assertTrue(module_stats["debugInfoIndexSavedToCache"])

        module_stats = self.get_debug_info_index_stats(obj_path)
        self.assertTrue(module_stats["debugInfoIndexLoadedFromCache"])
