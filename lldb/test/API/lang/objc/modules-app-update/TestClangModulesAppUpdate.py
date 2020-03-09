
import unittest2
import os
import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestClangModuleAppUpdate(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @skipIf(debug_info=no_match(["gmodules"]))
    def test_rebuild_app_modules_untouched(self):
        with open(self.getBuildArtifact("module.modulemap"), "w") as f:
            f.write("""
                    module Foo { header "f.h" }
                    """)
        with open(self.getBuildArtifact("f.h"), "w") as f:
            f.write("""
                    @import Foundation;
                    @interface Foo : NSObject {
                       int i;
                    }
                    +(instancetype)init;
                    @end
                    """)

        mod_cache = self.getBuildArtifact("private-module-cache")
        import os
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)
        self.build()
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")

        target, process, _, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.m"))
        bar = target.FindTypes('Bar').GetTypeAtIndex(0)
        foo = bar.GetDirectBaseClassAtIndex(0).GetType()
        self.assertEqual(foo.GetNumberOfFields(), 1)
        self.assertEqual(foo.GetFieldAtIndex(0).GetName(), "i")

        # Rebuild.
        process.Kill()
        os.remove(self.getBuildArtifact('main.o'))
        os.remove(self.getBuildArtifact('a.out'))
        self.build()

        # Reattach.
        target, process, _, _ = lldbutil.run_to_breakpoint_do_run(self, target, bkpt)
        bar = target.FindTypes('Bar').GetTypeAtIndex(0)
        foo = bar.GetDirectBaseClassAtIndex(0).GetType()
        self.assertEqual(foo.GetNumberOfFields(), 1)
        self.assertEqual(foo.GetFieldAtIndex(0).GetName(), "i")
