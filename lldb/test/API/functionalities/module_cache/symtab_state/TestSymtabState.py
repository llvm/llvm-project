"""Test SBModule.GetSymtabState() with the LLDB index cache."""

import glob
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os


@requireNotWasm("modules carry no build ID to cache on")
class SymtabStateTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False

    def get_module_cache_files(self, basename):
        module_file_glob = os.path.join(
            self.cache_dir, "llvmcache-*%s*-symtab-*" % (basename)
        )
        return glob.glob(module_file_glob)

    @no_debug_info_test
    @skipIfWindows
    def test(self):
        """
        Test that SBModule.GetSymtabState() reports whether the symbol table
        was parsed from the object file or loaded from the index cache.
        """
        # The index cache path is latched on first use, so this must be the
        # only test in this file.
        self.cache_dir = os.path.join(self.getBuildDir(), "lldb-module-cache")
        self.runCmd(
            'settings set symbols.lldb-index-cache-path "%s"' % (self.cache_dir)
        )
        self.runCmd("settings set symbols.enable-lldb-index-cache true")
        self.build()

        target = self.createTestTarget(load_dependent_modules=False)
        main_module = target.GetModuleAtIndex(0)
        self.assertTrue(main_module.IsValid())
        main_module.GetNumSymbols()
        self.assertTrue(
            main_module.GetSymtabState() & lldb.eSymtabStateParsedFromObjectFile
        )
        self.assertTrue(
            main_module.symtab_state & lldb.eSymtabStateParsedFromObjectFile
        )
        self.assertEqual(len(self.get_module_cache_files("a.out")), 1)

        # Release the module so the next target creates a new one that loads
        # its symbol table from the cache.
        del main_module
        self.dbg.DeleteTarget(target)
        del target
        state = self.get_symtab_state_in_new_target()
        self.assertTrue(state & lldb.eSymtabStateLoadedFromCache)

        # A cache file that can't be decoded must be replaced, not left in
        # place where it would keep the symbol table from ever being cached.
        cache_file = self.get_module_cache_files("a.out")[0]
        with open(cache_file, "r+b") as f:
            f.truncate(os.path.getsize(cache_file) // 2)
        state = self.get_symtab_state_in_new_target()
        self.assertTrue(state & lldb.eSymtabStateParsedFromObjectFile)
        self.assertTrue(state & lldb.eSymtabStateSavedToCache)
        state = self.get_symtab_state_in_new_target()
        self.assertTrue(state & lldb.eSymtabStateLoadedFromCache)

    def get_symtab_state_in_new_target(self):
        """
        Return the main module's symtab state from a newly created target,
        after releasing the modules of any previous target.
        """
        lldb.SBModule.GarbageCollectAllocatedModules()
        target = self.createTestTarget(load_dependent_modules=False)
        main_module = target.GetModuleAtIndex(0)
        self.assertTrue(main_module.IsValid())
        main_module.GetNumSymbols()
        state = main_module.GetSymtabState()
        del main_module
        self.dbg.DeleteTarget(target)
        return state
