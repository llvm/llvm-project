"""Test that symbol tables of binaries read out of a corefile's memory
are not saved in the LLDB index cache."""

import glob
import os
import shutil
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ModuleCacheCorefileMemoryModuleTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        TestBase.setUp(self)
        # Set the cache path before anything can touch the index cache,
        # which latches the path on first use.
        self.cache_dir = os.path.join(self.getBuildDir(), "lldb-module-cache")
        self.runCmd(
            'settings set -g symbols.lldb-index-cache-path "%s"' % self.cache_dir
        )

    def get_symtab_cache_files(self, basename):
        module_file_glob = os.path.join(
            self.cache_dir, "llvmcache-*%s*-symtab-*" % basename
        )
        return glob.glob(module_file_glob)

    @requireDarwin
    @skipIfRemote
    def test_corefile_memory_module_not_cached(self):
        """Create a corefile, then remove the binary from disk so lldb
        must read it out of the corefile's memory. With caching of memory
        modules enabled, confirm that no symbol table for the binary is
        written to the index cache."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        corefile = self.getBuildArtifact("process.core")

        # Enable the index cache for the live session as a positive control:
        # it shows that this cache path and glob do find the binary's symbol
        # table when it is cached.
        self.runCmd("settings set -g symbols.enable-lldb-index-cache true")

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        exe_module = target.FindModule(lldb.SBFileSpec("a.out"))
        self.assertGreater(exe_module.GetNumSymbols(), 0)
        cache_files = self.get_symtab_cache_files("a.out")
        self.assertEqual(
            len(cache_files), 1, "expected a symtab cache file for the live binary"
        )
        for cache_file in cache_files:
            os.unlink(cache_file)

        # Save a stack-only corefile, plus the binary's __TEXT and
        # __LINKEDIT so it can be read from the corefile's memory. A
        # full corefile would include the entire shared cache.
        options = lldb.SBSaveCoreOptions()
        options.SetOutputFile(lldb.SBFileSpec(corefile))
        options.SetStyle(lldb.eSaveCoreStackOnly)
        for segname in ["__TEXT", "__LINKEDIT"]:
            segment = exe_module.FindSection(segname)
            self.assertTrue(segment.IsValid())
            region = lldb.SBMemoryRegionInfo()
            self.assertSuccess(
                process.GetMemoryRegionInfo(segment.GetLoadAddress(target), region)
            )
            self.assertSuccess(options.AddMemoryRegionToSave(region))
        self.assertSuccess(process.SaveCore(options))
        self.assertTrue(os.path.isfile(corefile))

        self.assertSuccess(process.Kill())
        self.dbg.DeleteTarget(target)
        self.assertEqual(self.dbg.GetNumTargets(), 0)
        self.dbg.MemoryPressureDetected()

        # Remove the binary and its dSYM so lldb can only get the binary
        # from the corefile's memory.
        os.unlink(exe)
        dsym = exe + ".dSYM"
        if os.path.exists(dsym):
            shutil.rmtree(dsym)

        self.runCmd("settings set -g symbols.enable-lldb-index-cache false")
        self.runCmd(
            "settings set -g symbols.enable-lldb-index-cache-memory-modules true"
        )

        target = self.dbg.CreateTarget(None)
        self.assertTrue(target.IsValid())
        process = target.LoadCore(corefile)
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Confirm the binary was loaded in the corefile process.
        self.expect("image list", substrs=["a.out"])
        exe_module = target.FindModule(lldb.SBFileSpec("a.out"))
        self.assertTrue(exe_module.IsValid())

        # Force the symbol table to be created, which is when it would be
        # saved to the index cache.
        self.runCmd("bt")
        self.assertGreater(exe_module.GetNumSymbols(), 0)

        # Confirm no symbol table was saved in the index cache for the binary.
        self.assertEqual(len(self.get_symtab_cache_files("a.out")), 0)
