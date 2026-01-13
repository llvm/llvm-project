import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import gc
import os


class ReplaceDllTestCase(TestBase):
    @skipUnlessWindows
    def test(self):
        """
        Test that LLDB unlocks module files once all references are released.
        """

        exe = self.getBuildArtifact("a.out")
        foo = self.getBuildArtifact("foo.dll")
        bar = self.getBuildArtifact("bar.dll")

        self.build(
            dictionary={
                "DYLIB_NAME": "foo",
                "DYLIB_C_SOURCES": "foo.c",
                "C_SOURCES": "test.c",
            }
        )
        self.build(
            dictionary={
                "DYLIB_ONLY": "YES",
                "DYLIB_NAME": "bar",
                "DYLIB_C_SOURCES": "bar.c",
            }
        )

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        shlib_names = ["foo"]
        environment = self.registerSharedLibrariesWithTarget(target, shlib_names)
        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory()
        )
        self.assertEqual(process.GetExitStatus(), 42)

        module = next((m for m in target.modules if "foo" in m.file.basename), None)
        self.assertIsNotNone(module)
        self.assertEqual(module.file.fullpath, foo)

        target.RemoveModule(module)
        del module
        gc.collect()

        self.dbg.MemoryPressureDetected()

        os.remove(foo)
        os.rename(bar, foo)

        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory()
        )
        self.assertEqual(process.GetExitStatus(), 43)
