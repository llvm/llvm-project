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
    def test_kext_lookup_by_bundle_id_and_uuid(self):
        """Resolve a kext by bundle ID and UUID out of the darwin-kernel
        platform's index of the local filesystem."""

        # Restore the global state this test changes even if it fails partway.
        def cleanup():
            self.runCmd("platform select host")
            self.runCmd("settings clear platform.plugin.darwin-kernel.kext-directories")
            self.runCmd("settings clear symbols.enable-external-lookup")

        self.addTearDownHook(cleanup)

        # A kext is only indexed if it is a .kext bundle with a CFBundleID in
        # its Info.plist, and only the ones with a .dSYM sibling are searched.
        extensions = self.getBuildArtifact("Extensions")
        bundle = os.path.join(extensions, "mykext.kext")
        macos = os.path.join(bundle, "Contents", "MacOS")
        lldbutil.mkdir_p(macos)
        lldbutil.mkdir_p(bundle + ".dSYM")
        self.yaml2obj("mykext.yaml", os.path.join(macos, "mykext"))
        # The index only considers executable files inside the bundle.
        os.chmod(os.path.join(macos, "mykext"), 0o755)
        with open(os.path.join(bundle, "Contents", "Info.plist"), "w") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                '<plist version="1.0"><dict>\n'
                "  <key>CFBundleIdentifier</key><string>com.example.mykext</string>\n"
                "  <key>CFBundleExecutable</key><string>mykext</string>\n"
                "</dict></plist>\n"
            )

        # The index has to be the only thing that can answer, or a symbol server
        # could resolve the UUID and the test would pass without it.
        self.runCmd("settings set symbols.enable-external-lookup false")
        self.runCmd(
            "settings set platform.plugin.darwin-kernel.kext-directories " + extensions
        )
        self.runCmd("platform select darwin-kernel")

        target = self.dbg.CreateTarget("")
        self.assertTrue(target.IsValid())

        # The bundle ID travels as the file name, which is how the kext dynamic
        # loader asks for a kext.
        module = target.AddModule(
            "com.example.mykext", None, "17A97B33-09B7-3195-9408-DBD965D578A5"
        )
        self.assertTrue(module.IsValid(), "found the kext in the index")
        # Assert on the whole path, not just the file name: the shared module
        # list matches on UUID alone, so a module the other tests in this file
        # registered under a bare name would satisfy a file name check.
        self.assertEqual(module.GetFileSpec().fullpath, os.path.join(macos, "mykext"))

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
