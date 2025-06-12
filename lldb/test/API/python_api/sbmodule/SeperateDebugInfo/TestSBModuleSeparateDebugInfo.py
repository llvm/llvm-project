import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import os, signal, subprocess

from lldbsuite.test import lldbutil


class SBModuleSeparateDebugInfoCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.background_pid = None

    def tearDown(self):
        TestBase.tearDown(self)
        if self.background_pid:
            os.kill(self.background_pid, signal.SIGKILL)

    @skipIf(debug_info=no_match("dwo"))
    def test_get_separate_debug_info_files_dwo(self):
        """Test the SBModule::GetSeparateDebugInfoFiles"""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        # Target should have a DWO
        main_module = target.GetModuleAtIndex(0)
        file_specs = main_module.GetSeparateDebugInfoFiles()
        self.assertEqual(len(file_specs), 1)
        self.assertTrue(file_specs[0].GetFilename().endswith(".dwo"))

    @skipIf(debug_info=no_match("dsym"))
    def test_get_separate_debug_info_files_dsym(self):
        """Test the SBModule::GetSeparateDebugInfoFiles"""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        # Target should have a DWO
        main_module = target.GetModuleAtIndex(0)
        file_specs = main_module.GetSeparateDebugInfoFiles()
        self.assertEqual(len(file_specs), 1)
        self.assertTrue(file_specs[0].GetFilename().endswith(".a"))
