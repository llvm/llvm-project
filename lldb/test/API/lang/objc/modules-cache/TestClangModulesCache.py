"""Test that the clang modules cache directory can be controlled."""



import unittest2
import os
import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCModulesTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_expr(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.m")
        self.runCmd("settings set target.auto-import-clang-modules true")
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)
        self.assertFalse(os.path.isdir(mod_cache),
                         "module cache should not exist")
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"' % mod_cache)
        self.runCmd('settings set target.clang-module-search-paths "%s"'
                    % self.getSourceDir())
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", self.main_source_file)
        self.runCmd("expr @import Foo")
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")
