
import unittest2
import os
import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestClangModuleUpdate(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipIf(debug_info=no_match(["gmodules"]))
    @skipIfReproducer # VFS is a snapshot.
    @skipIfDarwin # rdar://76540904
    def test_expr(self):
        with open(self.getBuildArtifact("module.modulemap"), "w") as f:
            f.write("""
                    module Foo { header "f.h" }
                    """)
        with open(self.getBuildArtifact("f.h"), "w") as f:
            f.write("""
                    struct Q { int i; };
                    void f() {}
                    """)

        mod_cache = self.getBuildArtifact("private-module-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)
        d = {'OBJC_SOURCES': 'first.m'}
        self.build(dictionary=d)
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")

        logfile = self.getBuildArtifact("modules.log")
        self.runCmd("log enable -f %s lldb module" % logfile)
        target, process, _, bkpt = lldbutil.run_to_name_breakpoint(self, "main")
        self.assertIn("int i", str(target.FindTypes('Q').GetTypeAtIndex(0)))
        self.expect("image list -g", patterns=[r'first\.o', r'Foo.*\.pcm'])

        # Update the module.
        with open(self.getBuildArtifact("f.h"), "w") as f:
            f.write("""
                    struct S { int i; };
                    struct S getS() { struct S r = {1}; return r; }
                    void f() {}
                    """)

        # Rebuild.
        d = {'OBJC_SOURCES': 'second.m'}
        self.build(dictionary=d)

        # Reattach.
        process.Kill()
        target, process, _, _ = lldbutil.run_to_breakpoint_do_run(self, target, bkpt)
        self.assertIn("int i", str(target.FindTypes('S').GetTypeAtIndex(0)))
        self.expect("image list -g", patterns=[r'second\.o', r'Foo.*\.pcm'])

        # Check log file.
        found = False
        with open(logfile, 'r') as f:
            for line in f:
                if "module changed" in line and "Foo" in line:
                    found = True
        self.assertTrue(found)
