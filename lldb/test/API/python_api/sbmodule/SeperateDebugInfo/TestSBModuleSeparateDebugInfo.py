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
        """Test the SBModule::GetSeparateDebugInfoFiles with dwos"""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        # Target should have a DWO
        main_module = target.GetModuleAtIndex(0)
        module_specs = main_module.GetSeparateDebugInfoFiles()
        self.assertEqual(len(module_specs), 2)
        self.assertTrue(module_specs[0].GetFileSpec().GetFilename().endswith(".dwo"))

        filenames = [
            module_spec.GetFileSpec().GetFilename() for module_spec in module_specs
        ]
        self.assertTrue("bar.dwo" in filenames, filenames)
        self.assertTrue("main.dwo" in filenames, filenames)
        file_specs = [module_spec.GetFileSpec() for module_spec in module_specs]
        for file_spec in file_specs:
            self.assertTrue(file_spec.Exists())

    @skipIf(debug_info=no_match("dwp"))
    def test_get_separate_debug_info_files_dwp(self):
        """Test the SBModule::GetSeparateDebugInfoFiles with dwps"""
        self.build(debug_info="dwp")
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        # Target should have a DWO
        main_module = target.GetModuleAtIndex(0)
        module_specs = main_module.GetSeparateDebugInfoFiles()
        self.assertEqual(len(module_specs), 1)
        self.assertTrue(module_specs[0].GetFileSpec().GetFilename().endswith(".dwp"))
        self.assertTrue(module_specs[0].GetFileSpec().Exists())

    @no_debug_info_test
    @skipUnlessPlatform(["darwin"])
    def test_darwin_oso(self):
        """DWARF in .o files (no dSYM) -- returns .o file paths."""
        self.build(debug_info="dwarf")
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())

        files = module.GetSeparateDebugInfoFiles()
        self.assertGreaterEqual(
            files.GetSize(),
            2,
            "Expected at least 2 .o files (main + bar)",
        )

        o_paths = []
        for i in range(files.GetSize()):
            path = files.GetSpecAtIndex(i).GetFileSpec().fullpath
            self.assertTrue(path.endswith(".o"), f"Expected .o, got: {path}")
            o_paths.append(path)

        basenames = [os.path.basename(p) for p in o_paths]
        self.assertTrue(
            any("main" in b for b in basenames),
            f"Expected a .o for main, got: {basenames}",
        )
        self.assertTrue(
            any("bar" in b for b in basenames),
            f"Expected a .o for foo, got: {basenames}",
        )

        for p in o_paths:
            self.assertTrue(os.path.exists(p), f".o file should exist: {p}")
