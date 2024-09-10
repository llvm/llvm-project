"""
Test that LLDB can emit JIT objects when the appropriate setting is enabled
"""


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SaveJITObjectsTestCase(TestBase):
    def enumerateJITFiles(self):
        return [f for f in os.listdir(self.getBuildDir()) if f.startswith("jit")]

    def countJITFiles(self):
        return len(self.enumerateJITFiles())

    def cleanJITFiles(self):
        for j in self.enumerateJITFiles():
            os.remove(j)
        return

    @expectedFailureAll(oslist=["windows"])
    def test_save_jit_objects(self):
        self.build()
        os.chdir(self.getBuildDir())
        src_file = "main.c"
        src_file_spec = lldb.SBFileSpec(src_file)

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break", src_file_spec
        )

        frame = thread.frames[0]

        self.cleanJITFiles()
        frame.EvaluateExpression("(void*)malloc(0x1)")
        self.assertEqual(
            self.countJITFiles(), 0, "No files emitted with save-jit-objects-dir empty"
        )

        self.runCmd(
            "settings set target.save-jit-objects-dir {0}".format(self.getBuildDir())
        )
        frame.EvaluateExpression("(void*)malloc(0x1)")
        jit_files_count = self.countJITFiles()
        self.cleanJITFiles()
        self.assertNotEqual(
            jit_files_count,
            0,
            "At least one file emitted with save-jit-objects-dir set to the build dir",
        )

        process.Kill()
        os.chdir(self.getSourceDir())
