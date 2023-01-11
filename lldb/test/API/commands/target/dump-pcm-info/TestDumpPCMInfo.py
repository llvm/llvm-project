"""
Test 'target modules dump pcm-info'.
"""

import os
import shutil
import glob

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    @skipUnlessDarwin
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "return", lldb.SBFileSpec("main.m"))

        mod_cache = self.getBuildArtifact("private-module-cache")
        if os.path.isdir(mod_cache):
            shutil.rmtree(mod_cache)

        self.runCmd(f"settings set symbols.clang-modules-cache-path '{mod_cache}'")

        # Cause lldb to generate a Darwin-*.pcm
        self.runCmd("expression @import Darwin")

        # root/<config-hash>/<module-name>-<modulemap-path-hash>.pcm
        pcm_paths = glob.glob(os.path.join(mod_cache, '*', 'Darwin-*.pcm'))
        self.assertEqual(len(pcm_paths), 1, "Expected one Darwin pcm")
        pcm_path = pcm_paths[0]

        self.expect(
            f"target modules dump pcm-info '{pcm_path}'",
            startstr=f"Information for module file '{pcm_path}'",
            substrs=["Module name: Darwin"])
