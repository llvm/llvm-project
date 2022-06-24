
import unittest2
import os
import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestClangModuleHashMismatch(TestBase):

    @skipIf(debug_info=no_match(["gmodules"]))
    def test_expr(self):
        with open(self.getBuildArtifact("module.modulemap"), "w") as f:
            f.write("""
                    module Foo { header "f.h" }
                    """)
        with open(self.getBuildArtifact("f.h"), "w") as f:
            f.write("""
                    typedef int my_int;
                    void f() {}
                    """)

        mod_cache = self.getBuildArtifact("private-module-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)
        self.build()
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")

        logfile = self.getBuildArtifact("host.log")
        with open(logfile, 'w') as f:
            sbf = lldb.SBFile(f.fileno(), 'w', False)
            status = self.dbg.SetErrorFile(sbf)
            self.assertSuccess(status)

            target, _, _, _ = lldbutil.run_to_source_breakpoint(
                self, "break here", lldb.SBFileSpec("main.m"))
            target.GetModuleAtIndex(0).FindTypes('my_int')

        with open(logfile, 'r') as f:
            for line in f:
                if "hash mismatch" in line and "Foo" in line:
                    found = True
        self.assertTrue(found)
