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

        # No installed platform claims AMDGPU, so retain the selected host
        # platform while creating a target with the requested architecture.
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_PLATFORM", "")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_TRIPLE", "amdgpu-amd-amdhsa--gfx942")

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

    def create_library(self):
        self.build()

        library_path = self.getBuildArtifact("accelerator_lib.so")
        self.yaml2obj(self.getSourcePath("../accelerator_lib.yaml"), library_path)
        return library_path

    def launch_native(self):
        native_target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(native_target, VALID_TARGET)

        native_process = native_target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertTrue(native_process, PROCESS_IS_VALID)
        self.assertState(native_process.GetState(), lldb.eStateStopped)
        return native_target, native_process

    def write_native_image(self, native_target, native_process, image):
        symbols = native_target.FindSymbols("mock_gpu_accelerator_image")
        self.assertEqual(symbols.GetSize(), 1)
        symbol = symbols.GetContextAtIndex(0).GetSymbol()
        image_address = symbol.GetStartAddress().GetLoadAddress(native_target)
        self.assertNotEqual(image_address, lldb.LLDB_INVALID_ADDRESS)

        error = lldb.SBError()
        bytes_written = native_process.WriteMemory(image_address, image, error)
        self.assertSuccess(error)
        self.assertEqual(bytes_written, len(image))

    def connect_accelerator(self, native_target, native_process, dyld_connection):
        # Continuing to the connection hook synchronously creates the
        # accelerator target and queries for its loaded libraries.
        native_process.Continue()
        self.assertState(native_process.GetState(), lldb.eStateStopped)

        accelerator_targets = [
            candidate for candidate in self.dbg if candidate != native_target
        ]
        self.assertEqual(len(accelerator_targets), 1)
        accelerator_target = accelerator_targets[0]
        self.assertTrue(accelerator_target.IsValid())
        accelerator_process = accelerator_target.GetProcess()
        self.assertTrue(accelerator_process.IsValid())
        self.assertState(accelerator_process.GetState(), lldb.eStateStopped)

        self.dbg.SetSelectedTarget(accelerator_target)
        self.expect(
            "process plugin packet send jLLDBSettings",
            substrs=[
                '"accelerator_plugin_name":"mock"',
                '"dyld_plugin_name":"accelerator-gdb-remote"',
                '"send_dyld_packet_to_accelerator":'
                + ("true" if dyld_connection == "accelerator" else "false"),
            ],
        )
        return accelerator_target

    def assert_library_loaded(self, target, path, expected_text_address):
        library_basename = os.path.basename(path)
        module = None
        for candidate in target.module_iter():
            if candidate.GetFileSpec().GetFilename() == library_basename:
                module = candidate
                break
        self.assertIsNotNone(module, "accelerator library should be loaded")

        text = module.FindSection(".text")
        self.assertTrue(text.IsValid())
        self.assertEqual(text.GetLoadAddress(target), expected_text_address)

    @skipIfRemote
    @skipIfDarwin  # qProcessInfo cannot encode an AMDGPU architecture on Darwin.
    @add_test_categories(["llgs"])
    def test_library_from_accelerator_process(self):
        """The accelerator loader queries the second LLGS process directly."""
        library_path = self.create_library()
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_DYLD_CONNECTION", "accelerator")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_LIBRARY_PATH", library_path)

        native_target, native_process = self.launch_native()
        accelerator_target = self.connect_accelerator(
            native_target, native_process, "accelerator"
        )
        self.assert_library_loaded(accelerator_target, library_path, 0x10001000)

    @skipIfRemote
    @skipIfDarwin  # qProcessInfo cannot encode an AMDGPU architecture on Darwin.
    @add_test_categories(["llgs"])
    def test_dynamic_loader_stop_from_accelerator_process(self):
        """The mock accelerator reports a dynamic-loader stop through LLGS."""
        library_path = self.create_library()
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_DYLD_CONNECTION", "accelerator")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_LIBRARY_PATH", library_path)
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_DYNAMIC_LOADER_STOP_ON_RESUME", "1")

        native_target, native_process = self.launch_native()
        accelerator_target = self.connect_accelerator(
            native_target, native_process, "accelerator"
        )
        self.runCmd("settings set target.process.stop-on-sharedlibrary-events true")
        self.addTearDownHook(
            lambda: self.runCmd(
                "settings set target.process.stop-on-sharedlibrary-events false"
            )
        )
        accelerator_process = accelerator_target.GetProcess()

        self.assertSuccess(accelerator_process.Continue())

        self.assertState(accelerator_process.GetState(), lldb.eStateStopped)
        self.assertStopReason(
            accelerator_process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonDynamicLoader,
        )

    @skipIfRemote
    @skipIfDarwin  # qProcessInfo cannot encode an AMDGPU architecture on Darwin.
    @add_test_categories(["llgs"])
    def test_library_from_native_plugin(self):
        """The accelerator loader routes through the native LLGS plugin."""
        library_path = self.create_library()
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_DYLD_CONNECTION", "native")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_NATIVE_LIBRARY_PATH", library_path)

        native_target, native_process = self.launch_native()
        accelerator_target = self.connect_accelerator(
            native_target, native_process, "native"
        )
        self.assert_library_loaded(accelerator_target, library_path, 0x7F001000)

    @skipIfRemote
    @skipIfDarwin  # qProcessInfo cannot encode an AMDGPU architecture on Darwin.
    @add_test_categories(["llgs"])
    def test_library_from_native_memory(self):
        """An image in native memory is loaded over the accelerator connection."""
        library_path = self.create_library()
        with open(library_path, "rb") as library_file:
            library_data = library_file.read()

        reported_path = self.getBuildArtifact("missing-accelerator-lib.so")
        self.assertFalse(os.path.exists(reported_path))
        self.assertLessEqual(len(library_data), 8192)
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_DYLD_CONNECTION", "accelerator")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_LIBRARY_PATH", reported_path)
        self.set_mock_env(
            "LLDB_MOCK_ACCELERATOR_NATIVE_MEMORY_SIZE", str(len(library_data))
        )

        native_target, native_process = self.launch_native()
        self.write_native_image(native_target, native_process, library_data)
        accelerator_target = self.connect_accelerator(
            native_target, native_process, "accelerator"
        )
        self.assert_library_loaded(accelerator_target, reported_path, 0x10001000)
