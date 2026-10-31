"""Test that DataFileCache preserves re-export symbols correctly."""

import glob
import subprocess
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os


class DataFileCacheReexportSymbolsTest(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    @requireDarwin
    def test_reexport_symbols_in_datafilecache(self):
        """
        Test that DataFileCache correctly encodes and decodes
        re-export symbols in Mach-O binaries.
        """
        self.build()
        source_file = lldb.SBFileSpec("main.c")
        exe = os.path.join(self.getBuildDir(), "a.out")

        #
        # First launch:  No DataFileCache in use.
        #
        self.runCmd("settings set -g symbols.enable-lldb-index-cache false")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", source_file
        )
        ci = self.dbg.GetCommandInterpreter()
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand("target modules dump symtab libsystem_c.dylib", res)
        self.assertTrue(res.Succeeded())
        actual_binary_symtab = lldb.SBStream()
        res.GetDescription(actual_binary_symtab)

        process.Kill()
        self.dbg.DeleteTarget(target)
        self.assertEqual(self.dbg.GetNumTargets(), 0)
        self.dbg.MemoryPressureDetected()

        # Set the lldb-index-cache settings in the global settings
        # that new Targets will inherit.
        cache_dir = os.path.join(self.getBuildDir(), "lldb-module-cache")
        self.runCmd('settings set -g symbols.lldb-index-cache-path "%s"' % (cache_dir))
        self.runCmd("settings set -g symbols.enable-lldb-index-cache true")

        #
        # Second launch: Create DataFileCaches as we load the binaries.
        #
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", source_file
        )
        if self.TraceOn():
            print("ls -l %s" % cache_dir)
            subprocess.call(["ls", "-l", cache_dir])
        module_file_glob = os.path.join(
            cache_dir, "llvmcache-*libsystem_c.dylib*-symtab-*"
        )
        self.assertEqual(len(glob.glob(module_file_glob)), 1)

        process.Kill()
        self.dbg.DeleteTarget(target)
        self.assertEqual(self.dbg.GetNumTargets(), 0)
        self.dbg.MemoryPressureDetected()

        #
        # Third launch: Load binaries with their symbol tables from DataFileCache.
        #
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", source_file
        )
        if self.TraceOn():
            print("ls -l %s" % cache_dir)
            subprocess.call(["ls", "-l", cache_dir])
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand("target modules dump symtab libsystem_c.dylib", res)
        self.assertTrue(res.Succeeded())
        datafilecache_symtab = lldb.SBStream()
        res.GetDescription(datafilecache_symtab)

        # Compare the libsystem_c.dylib symtab directly from the Mach-O binary
        # to the symtab decoded from the DataFileCache.  They must be identical.
        self.assertEqual(actual_binary_symtab.GetData(), datafilecache_symtab.GetData())
