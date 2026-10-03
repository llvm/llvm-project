"""End-to-end test for the mock accelerator's dynamic loader."""

import os

import lldb
from lldbsuite.test import configuration
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class MockAcceleratorDynamicLoaderTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super().setUp()
        if "mock-accelerator" not in configuration.enabled_plugins:
            self.skipTest("mock-accelerator plugin is not enabled")

    def set_mock_env(self, name, value):
        """Set an environment variable inherited by lldb-server."""
        previous = os.environ.get(name)
        os.environ[name] = value

        def restore():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

        self.addTearDownHook(restore)

    def accelerator_target(self, native_target):
        for candidate in self.dbg:
            if candidate != native_target:
                return candidate
        return None

    @skipIfRemote
    @skipIfDarwin  # qProcessInfo cannot encode an AMDGPU architecture on Darwin.
    @add_test_categories(["llgs"])
    def test_library_from_accelerator_process(self):
        """The accelerator loader queries the second LLGS process directly."""
        self.build()

        library_path = self.getBuildArtifact("accelerator_lib.so")
        self.yaml2obj("accelerator_lib.yaml", library_path)

        # No installed platform claims AMDGPU, so retain the selected host
        # platform while creating a target with the requested architecture.
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_PLATFORM", "")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_TRIPLE", "amdgpu-amd-amdhsa--gfx942")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_LIBRARY_PATH", library_path)

        # This test does not exercise instruction decoding. Avoid asking the
        # generic AMDGPU target to disassemble its synthetic stop PC.
        self.runCmd("settings set stop-disassembly-display never")

        native_target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(native_target, VALID_TARGET)

        # The first stop initializes the plugin. Continuing to its connection
        # hook synchronously creates the accelerator target and queries its
        # ProcessMockAccelerator for loaded libraries.
        native_process = native_target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertTrue(native_process, PROCESS_IS_VALID)
        self.assertState(native_process.GetState(), lldb.eStateStopped)
        native_process.Continue()
        self.assertState(native_process.GetState(), lldb.eStateStopped)

        self.assertEqual(self.dbg.GetNumTargets(), 2)
        accelerator_target = self.accelerator_target(native_target)
        self.assertIsNotNone(accelerator_target)
        self.assertTrue(accelerator_target.IsValid())
        accelerator_process = accelerator_target.GetProcess()
        self.assertTrue(accelerator_process.IsValid())
        self.assertState(accelerator_process.GetState(), lldb.eStateStopped)

        module = None
        library_basename = os.path.basename(library_path)
        for candidate in accelerator_target.module_iter():
            if candidate.GetFileSpec().GetFilename() == library_basename:
                module = candidate
                break
        self.assertIsNotNone(module, "accelerator library should be loaded")

        text = module.FindSection(".text")
        self.assertTrue(text.IsValid())
        self.assertEqual(text.GetLoadAddress(accelerator_target), 0x10001000)
