"""Test that symbols.lldb-index-cache.force-disable stops all caching."""

import glob
import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireNotWasm("modules carry no build ID to cache on")
class ModuleCacheForceDisableTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def get_symtab_cache_files(self, basename):
        return glob.glob(
            os.path.join(self.cache_dir, "llvmcache-*%s*-symtab-*" % basename)
        )

    def parse_symtab(self, exe_name):
        target = self.createTestTarget(
            file_path=self.getBuildArtifact(exe_name), load_dependent_modules=False
        )
        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())
        # Parsing the symbol table caches it, if caching is enabled.
        module.GetNumSymbols()

    @skipIfWindows
    def test(self):
        # Two copies of the same program, so that each one's symbol table is
        # parsed for the first time with a different setting.
        self.build(dictionary={"EXE": "a.out"})
        self.build(dictionary={"EXE": "b.out"})

        self.cache_dir = os.path.join(self.getBuildDir(), "lldb-module-cache")
        self.runCmd('settings set symbols.lldb-index-cache.path "%s"' % self.cache_dir)
        self.runCmd("settings set symbols.lldb-index-cache.type.symtab true")

        self.runCmd("settings set symbols.lldb-index-cache.force-disable true")
        self.parse_symtab("a.out")
        self.assertFalse(
            os.path.exists(self.cache_dir),
            "the cache directory isn't created while caching is force-disabled",
        )

        # The same settings do cache the symbol table without force-disable.
        self.runCmd("settings set symbols.lldb-index-cache.force-disable false")
        self.parse_symtab("b.out")
        self.assertEqual(
            len(self.get_symtab_cache_files("b.out")),
            1,
            "the symbol table is cached once force-disable is turned off",
        )
        self.assertEqual(
            len(self.get_symtab_cache_files("a.out")),
            0,
            "the symbol table parsed while force-disabled wasn't cached",
        )
