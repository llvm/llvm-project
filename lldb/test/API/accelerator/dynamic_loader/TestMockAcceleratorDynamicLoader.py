"""
End-to-end test for the accelerator dynamic loader.

When the client creates and connects the accelerator target (triggered by the
mock accelerator plugin's connection hook), that target selects the
"accelerator-gdb-remote" dynamic loader via the jLLDBSettings packet, instead of
auto-selecting a loader from the triple. The loader then asks the accelerator
GDB server for its loaded libraries via
jAcceleratorPluginGetDynamicLoaderLibraryInfo and loads them into the target.

This verifies two of the ways a library can be provided:
  1. As a whole shared object on disk.
  3. As a shared object embedded in a containing file, located by file offset
     and size (as a library added to a container with llvm-objcopy would be).

The in-memory case (2) needs host support that is out of scope here, so it is
not exercised.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration


class MockAcceleratorDynamicLoaderTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super().setUp()
        if "mock-accelerator" not in configuration.enabled_plugins:
            self.skipTest("mock-accelerator plugin is not enabled")

    def set_mock_env(self, name, value):
        """Set an environment variable the mock plugin reads (it is inherited by
        the lldb-server that hosts the plugin), restoring it after the test."""
        previous = os.environ.get(name)
        os.environ[name] = value

        def restore():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

        self.addTearDownHook(restore)

    def make_container(self, lib_path):
        """Embed the library bytes in a larger container file, returning the
        container path plus the (offset, size) of the embedded object."""
        with open(lib_path, "rb") as f:
            lib_bytes = f.read()
        # Model a library embedded in a containing object file (as llvm-objcopy
        # would add it to an executable): the container is itself a valid object
        # file, with the embedded library located deeper in the file by
        # file_offset/file_size. We reuse the library bytes as the outer object
        # and page-align the embedded copy.
        prefix = lib_bytes
        pad = (-len(prefix)) % 0x1000
        prefix += b"\x00" * pad
        container_path = self.getBuildArtifact("container.bin")
        with open(container_path, "wb") as f:
            f.write(prefix)
            f.write(lib_bytes)
        return container_path, len(prefix), len(lib_bytes)

    def find_module(self, target, basename):
        for i in range(target.GetNumModules()):
            module = target.GetModuleAtIndex(i)
            if module.GetFileSpec().GetFilename() == basename:
                return module
        return None

    def assert_loaded_at(self, target, module, base):
        """The module's .text must resolve to a load address at or above the
        base the loader slid it to, proving the load address was applied."""
        section = module.FindSection(".text")
        self.assertTrue(section.IsValid(), "library should have a .text section")
        load_addr = section.GetLoadAddress(target)
        self.assertNotEqual(
            load_addr,
            lldb.LLDB_INVALID_ADDRESS,
            "library section should have a load address in the accelerator target",
        )
        self.assertGreaterEqual(load_addr, base)

    @skipIfRemote
    @add_test_categories(["llgs"])
    def test_accelerator_dynamic_loader(self):
        """The accelerator target loads the libraries reported by the server."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        lib_ondisk = self.getBuildArtifact("libgpu_lib.so")
        container, offset, size = self.make_container(lib_ondisk)

        # Tell the mock accelerator process which libraries to report, and where.
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_LIB_ONDISK", lib_ondisk)
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_LIB_CONTAINER", container)
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_LIB_OFFSET", str(offset))
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_LIB_SIZE", str(size))

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch stops at the plugin's initialize breakpoint; continuing reaches
        # the connection hook that creates the accelerator target.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertState(process.GetState(), lldb.eStateStopped)
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)

        # The accelerator target now exists alongside the native target.
        self.assertEqual(self.dbg.GetNumTargets(), 2)
        accelerator_target = None
        for i in range(self.dbg.GetNumTargets()):
            candidate = self.dbg.GetTargetAtIndex(i)
            if candidate != target:
                accelerator_target = candidate
                break
        self.assertTrue(accelerator_target.IsValid())

        # Scenario 1: the whole-file shared library was loaded.
        ondisk_module = self.find_module(accelerator_target, "libgpu_lib.so")
        self.assertIsNotNone(
            ondisk_module,
            "on-disk library should be loaded into the accelerator target",
        )
        self.assert_loaded_at(accelerator_target, ondisk_module, 0x10000000)

        # Scenario 3: the embedded shared library, located by file offset/size,
        # was loaded as a distinct module from its container file.
        embedded_module = self.find_module(accelerator_target, "container.bin")
        self.assertIsNotNone(
            embedded_module,
            "embedded library should be loaded into the accelerator target",
        )
        self.assert_loaded_at(accelerator_target, embedded_module, 0x20000000)
