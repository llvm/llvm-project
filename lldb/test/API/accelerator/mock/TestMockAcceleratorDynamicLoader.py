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

    def run_library_test(self, dyld_connection, use_native_memory=False):
        self.build()

        library_path = self.getBuildArtifact("accelerator_lib.so")
        self.yaml2obj("accelerator_lib.yaml", library_path)
        with open(library_path, "rb") as library_file:
            library_data = library_file.read()

        # No installed platform claims AMDGPU, so retain the selected host
        # platform while creating a target with the requested architecture.
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_PLATFORM", "")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_TRIPLE", "amdgpu-amd-amdhsa--gfx942")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_DYLD_CONNECTION", dyld_connection)
        reported_path = library_path
        if dyld_connection == "accelerator":
            if use_native_memory:
                reported_path = self.getBuildArtifact("missing-accelerator-lib.so")
                self.assertFalse(os.path.exists(reported_path))
                self.assertLessEqual(len(library_data), 8192)
                self.set_mock_env(
                    "LLDB_MOCK_ACCELERATOR_NATIVE_MEMORY_SIZE",
                    str(len(library_data)),
                )
            self.set_mock_env("LLDB_MOCK_ACCELERATOR_LIBRARY_PATH", reported_path)
        else:
            self.set_mock_env("LLDB_MOCK_ACCELERATOR_NATIVE_LIBRARY_PATH", library_path)

        # This test does not exercise instruction decoding. Avoid asking the
        # generic AMDGPU target to disassemble its synthetic stop PC.
        self.runCmd("settings set stop-disassembly-display never")

        native_target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(native_target, VALID_TARGET)

        # The first stop initializes the plugin. Continuing to its connection
        # hook synchronously creates the accelerator target and queries for its
        # loaded libraries using the selected connection.
        native_process = native_target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertTrue(native_process, PROCESS_IS_VALID)
        self.assertState(native_process.GetState(), lldb.eStateStopped)
        if use_native_memory:
            symbols = native_target.FindSymbols("mock_gpu_accelerator_image")
            self.assertEqual(symbols.GetSize(), 1)
            symbol = symbols.GetContextAtIndex(0).GetSymbol()
            image_address = symbol.GetStartAddress().GetLoadAddress(native_target)
            self.assertNotEqual(image_address, lldb.LLDB_INVALID_ADDRESS)
            error = lldb.SBError()
            bytes_written = native_process.WriteMemory(
                image_address, library_data, error
            )
            self.assertSuccess(error)
            self.assertEqual(bytes_written, len(library_data))
        native_process.Continue()
        self.assertState(native_process.GetState(), lldb.eStateStopped)

        self.assertEqual(self.dbg.GetNumTargets(), 2)
        accelerator_target = self.accelerator_target(native_target)
        self.assertIsNotNone(accelerator_target)
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

        module = None
        library_basename = os.path.basename(reported_path)
        for candidate in accelerator_target.module_iter():
            if candidate.GetFileSpec().GetFilename() == library_basename:
                module = candidate
                break
        self.assertIsNotNone(module, "accelerator library should be loaded")

        text = module.FindSection(".text")
        self.assertTrue(text.IsValid())
        expected_load_address = (
            0x10001000 if dyld_connection == "accelerator" else 0x7F001000
        )
        self.assertEqual(text.GetLoadAddress(accelerator_target), expected_load_address)

    @skipIfRemote
    @skipIfDarwin  # qProcessInfo cannot encode an AMDGPU architecture on Darwin.
    @add_test_categories(["llgs"])
    def test_library_from_accelerator_process(self):
        """The accelerator loader queries the second LLGS process directly."""
        self.run_library_test("accelerator")

    @skipIfRemote
    @skipIfDarwin  # qProcessInfo cannot encode an AMDGPU architecture on Darwin.
    @add_test_categories(["llgs"])
    def test_library_from_native_plugin(self):
        """The accelerator loader routes through the native LLGS plugin."""
        self.run_library_test("native")

    @skipIfRemote
    @skipIfDarwin  # qProcessInfo cannot encode an AMDGPU architecture on Darwin.
    @add_test_categories(["llgs"])
    def test_library_from_native_memory(self):
        """An image in native memory is loaded over the accelerator connection."""
        self.run_library_test("accelerator", use_native_memory=True)
