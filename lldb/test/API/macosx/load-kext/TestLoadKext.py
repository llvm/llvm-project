"""
Test loading of a kext binary.
"""


import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LoadKextTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_load_kext(self):
        """Test that lldb can load a kext binary."""

        # Create kext from YAML.
        self.yaml2obj("mykext.yaml", self.getBuildArtifact("mykext"))

        target = self.dbg.CreateTarget(self.getBuildArtifact("mykext"))

        self.assertTrue(target.IsValid())

        self.assertEqual(target.GetNumModules(), 1)
        mod = target.GetModuleAtIndex(0)
        self.assertEqual(mod.GetFileSpec().GetFilename(), "mykext")

    @skipUnlessDarwin
    def test_kernel_lookup_by_uuid_without_a_process(self):
        """Search the darwin-kernel platform for a kernel before there is a
        process to search on behalf of."""

        # PlatformDarwinKernel only indexes a kernel binary that has a .dSYM
        # sibling, and only that branch goes looking for a symbol file.
        kernel_dir = self.getBuildArtifact("kernels")
        lldbutil.mkdir_p(kernel_dir)
        kernel = os.path.join(kernel_dir, "kernel.test")
        self.yaml2obj("mykext.yaml", kernel)
        lldbutil.mkdir_p(kernel + ".dSYM")

        self.runCmd(
            "settings set platform.plugin.darwin-kernel.kext-directories " + kernel_dir
        )
        self.runCmd("platform select darwin-kernel")

        def cleanup():
            self.runCmd("platform select host")
            self.runCmd("settings clear platform.plugin.darwin-kernel.kext-directories")

        self.addTearDownHook(cleanup)

        target = self.dbg.CreateTarget("")
        self.assertTrue(target.IsValid())

        # A UUID with no file name reaches the kernel branch of
        # PlatformDarwinKernel::GetSharedModule, which matches the binary above
        # and then searches for its symbols. There is no process here, and the
        # search must not need one.
        target.AddModule(None, None, "17A97B33-09B7-3195-9408-DBD965D578A5")
