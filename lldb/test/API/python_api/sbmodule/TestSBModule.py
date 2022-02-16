"""Test the SBDModule APIs."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import os, signal, subprocess

class SBModuleAPICase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.background_pid = None

    def tearDown(self):
        TestBase.tearDown(self)
        if self.background_pid:
            os.kill(self.background_pid, signal.SIGKILL)

    @skipUnlessDarwin
    def test_module_is_file_backed(self):
        """Test the SBModule::IsFileBacked() method"""
        self.build()
        target, _, _, _  = lldbutil.run_to_source_breakpoint(self, "// break here",
                                                    lldb.SBFileSpec("main.c"))

        self.assertGreater(target.GetNumModules(), 0)
        main_module = target.GetModuleAtIndex(0)
        self.assertEqual(main_module.GetFileSpec().GetFilename(), "a.out")
        self.assertTrue(main_module.IsFileBacked(),
                         "The module should be backed by a file on disk")

        self.dbg.DeleteTarget(target)
        self.assertEqual(self.dbg.GetNumTargets(), 0)

        exe = self.getBuildArtifact("a.out")
        background_process = subprocess.Popen([exe])
        self.assertTrue(background_process, "process is not valid")
        self.background_pid = background_process.pid
        os.unlink(exe)

        target = self.dbg.CreateTarget('')
        self.assertEqual(self.dbg.GetNumTargets(), 1)
        error = lldb.SBError()
        process = target.AttachToProcessWithID(self.dbg.GetListener(),
                                               self.background_pid, error)
        self.assertTrue(error.Success() and process,  PROCESS_IS_VALID)
        main_module = target.GetModuleAtIndex(0)
        self.assertEqual(main_module.GetFileSpec().GetFilename(), "a.out")
        self.assertFalse(main_module.IsFileBacked(),
                         "The module should not be backed by a file on disk.")

        error = process.Destroy()
        self.assertSuccess(error, "couldn't destroy process %s" % background_process.pid)

