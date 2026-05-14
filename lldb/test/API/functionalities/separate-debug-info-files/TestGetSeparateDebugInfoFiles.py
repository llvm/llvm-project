"""
Test SBModule.GetSeparateDebugInfoFiles() API.

Uses multiple source files (main.c, foo.c) to verify behavior with:
  1. No split DWARF (dwarf) -- returns empty list
  2. Split DWARF .dwo files (dwo) -- returns DWO file paths
  3. Split DWARF .dwp file (dwp) -- returns single DWP file path
  4. Darwin .o files, no dSYM (dwarf) -- returns .o file paths
  5. Darwin dSYM present (dsym) -- returns empty list
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGetSeparateDebugInfoFilesNoSplit(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    @skipIfWindows
    def test_no_split_dwarf(self):
        """No split DWARF -- GetSeparateDebugInfoFiles returns empty."""
        self.build(debug_info="dwarf")
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())

        files = module.GetSeparateDebugInfoFiles()
        self.assertEqual(
            files.GetSize(), 0,
            f"Expected no separate debug info files, got {files.GetSize()}",
        )


class TestGetSeparateDebugInfoFilesDwo(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    @skipUnlessPlatform(["linux", "freebsd"])
    def test_split_dwarf_dwo(self):
        """Split DWARF with .dwo files -- returns DWO file paths."""
        self.build(debug_info="dwo")
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())

        files = module.GetSeparateDebugInfoFiles()
        self.assertGreaterEqual(
            files.GetSize(), 2,
            "Expected at least 2 .dwo files (main + foo)",
        )

        dwo_paths = []
        for i in range(files.GetSize()):
            path = files.GetSpecAtIndex(i).GetFileSpec().fullpath
            self.assertTrue(path.endswith(".dwo"), f"Expected .dwo, got: {path}")
            dwo_paths.append(path)

        basenames = [os.path.basename(p) for p in dwo_paths]
        self.assertTrue(
            any("main" in b for b in basenames),
            f"Expected a .dwo for main, got: {basenames}",
        )
        self.assertTrue(
            any("foo" in b for b in basenames),
            f"Expected a .dwo for foo, got: {basenames}",
        )

        for p in dwo_paths:
            self.assertTrue(os.path.exists(p), f"DWO file should exist: {p}")


class TestGetSeparateDebugInfoFilesDwp(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    @skipUnlessPlatform(["linux", "freebsd"])
    def test_split_dwarf_dwp(self):
        """Split DWARF with .dwp file -- returns single DWP path."""
        self.build(debug_info="dwp")
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())

        files = module.GetSeparateDebugInfoFiles()
        self.assertEqual(
            files.GetSize(), 1,
            f"Expected 1 .dwp entry, got {files.GetSize()}",
        )
        path = files.GetSpecAtIndex(0).GetFileSpec().fullpath
        self.assertTrue(path.endswith(".dwp"), f"Expected .dwp, got: {path}")
        self.assertTrue(os.path.exists(path), f"DWP should exist: {path}")


class TestGetSeparateDebugInfoFilesDarwinOso(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

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
            files.GetSize(), 2,
            "Expected at least 2 .o files (main + foo)",
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
            any("foo" in b for b in basenames),
            f"Expected a .o for foo, got: {basenames}",
        )

        for p in o_paths:
            self.assertTrue(os.path.exists(p), f".o file should exist: {p}")


class TestGetSeparateDebugInfoFilesDarwinDsym(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    @skipUnlessPlatform(["darwin"])
    def test_darwin_dsym(self):
        """dSYM present -- returns empty list."""
        self.build(debug_info="dsym")
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())

        files = module.GetSeparateDebugInfoFiles()
        self.assertEqual(
            files.GetSize(), 0,
            f"Expected no separate debug info with dSYM, got {files.GetSize()}",
        )
