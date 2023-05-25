"""Test the LLDB module cache funcionality."""

import glob
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os
import time


class ModuleCacheTestcaseSimple(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number in a(int) to break at.
        self.cache_dir = os.path.join(self.getBuildDir(), "lldb-module-cache")
        # Set the lldb module cache directory to a directory inside the build
        # artifacts directory so no other tests are interfered with.
        self.runCmd(
            'settings set symbols.lldb-index-cache-path "%s"' % (self.cache_dir)
        )
        self.runCmd("settings set symbols.enable-lldb-index-cache true")
        self.build()

    def get_module_cache_files(self, basename):
        module_file_glob = os.path.join(
            self.cache_dir, "llvmcache-*%s*-symtab-*" % (basename)
        )
        return glob.glob(module_file_glob)

    # Doesn't depend on any specific debug information.
    @no_debug_info_test
    @skipIfWindows
    def test(self):
        """
        Test module cache functionality for a simple object file.

        This will test that if we enable the module cache, we have a
        corresponding index cache entry for the symbol table for the
        executable. It also removes the executable, rebuilds so that the
        modification time of the binary gets updated, and then creates a new
        target and should cause the cache to get updated so the cache file
        should get an updated modification time.
        """
        exe = self.getBuildArtifact("a.out")

        # Create a module with no dependencies.
        target = self.createTestTarget(load_dependent_modules=False)

        # Get the executable module and get the number of symbols to make
        # sure the symbol table gets parsed and cached. The module cache is
        # enabled in the setUp() function.
        main_module = target.GetModuleAtIndex(0)
        self.assertTrue(main_module.IsValid())
        # Make sure the symbol table gets loaded and cached
        main_module.GetNumSymbols()
        cache_files = self.get_module_cache_files("a.out")
        self.assertEqual(
            len(cache_files), 1, "make sure there is only one cache file for 'a.out'"
        )
        symtab_cache_path = cache_files[0]
        exe_mtime_1 = os.path.getmtime(exe)
        symtab_mtime_1 = os.path.getmtime(symtab_cache_path)
        # Now remove the executable and sleep for a few seconds to make sure we
        # get a different creation and modification time for the file since some
        # OSs store the modification time in seconds since Jan 1, 1970.
        os.remove(exe)
        self.assertEqual(
            os.path.exists(exe),
            False,
            "make sure we were able to remove the executable",
        )
        time.sleep(2)
        # Now rebuild the binary so it has a different content which should
        # update the UUID to make the cache miss when it tries to load the
        # symbol table from the binary at the same path.
        self.build(dictionary={"CFLAGS_EXTRAS": "-DEXTRA_FUNCTION"})
        self.assertEqual(
            os.path.exists(exe), True, "make sure executable exists after rebuild"
        )
        # Make sure the modification time has changed or this test will fail.
        exe_mtime_2 = os.path.getmtime(exe)
        self.assertNotEqual(
            exe_mtime_1,
            exe_mtime_2,
            "make sure the modification time of the executable has changed",
        )
        # Make sure the module cache still has an out of date cache with the
        # same old modification time.
        self.assertEqual(
            symtab_mtime_1,
            os.path.getmtime(symtab_cache_path),
            "check that the 'symtab' cache file modification time doesn't match the executable modification time after rebuild",
        )
        # Create a new target and get the symbols again, and make sure the cache
        # gets updated for the symbol table cache
        target = self.createTestTarget(load_dependent_modules=False)
        main_module = target.GetModuleAtIndex(0)
        self.assertTrue(main_module.IsValid())
        main_module.GetNumSymbols()
        self.assertEqual(
            os.path.exists(symtab_cache_path),
            True,
            'make sure "symtab" cache files exists after cache is updated',
        )
        symtab_mtime_2 = os.path.getmtime(symtab_cache_path)
        self.assertNotEqual(
            symtab_mtime_1,
            symtab_mtime_2,
            'make sure modification time of "symtab-..." changed',
        )
