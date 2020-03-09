
import unittest2
import os
import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestClangModuleHashMismatch(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
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
        self.runCmd("log enable -v -f %s lldb host" % logfile)
        target, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.m"))
        target.GetModuleAtIndex(0).FindTypes('my_int') 

        found = False
        with open(logfile, 'r') as f:
            for line in f:
                if "hash mismatch" in line and "Foo" in line:
                    found = True
        self.assertTrue(found)
