"""
Test the accelerator dynamic loader against a mock GDB server.

The loader is selected for accelerator architectures debugged over gdb-remote.
It then asks the server for the loaded libraries via
jAcceleratorPluginGetDynamicLoaderLibraryInfo and loads them into the target.
"""

import json
import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

DYLD_PACKET = "jAcceleratorPluginGetDynamicLoaderLibraryInfo:"


class AcceleratorResponder(MockGDBServerResponder):
    """Serves a fixed set of library infos, and counts how often it is asked."""

    def __init__(self, library_infos):
        MockGDBServerResponder.__init__(self)
        self.library_infos = library_infos
        self.dyld_queries = 0

    def qSupported(self, client_supported):
        return super().qSupported(client_supported) + ";qXfer:features:read+"

    def qXferRead(self, obj, annex, offset, length):
        # An accelerator architecture has no built-in register set in lldb.
        if obj == "features" and annex == "target.xml":
            return (
                """<?xml version="1.0"?>
                <target version="1.0">
                  <feature name="org.llvm.accelerator">
                    <reg name="pc" bitsize="64" regnum="0" type="code_ptr" group="general"/>
                  </feature>
                </target>""",
                False,
            )
        return None, False

    def readRegisters(self):
        return "00" * 8

    def other(self, packet):
        if packet.startswith(DYLD_PACKET):
            self.dyld_queries += 1
            # "}" is the gdb-remote escape character.
            return escape_binary(
                json.dumps({"library_infos": self.library_infos}, separators=(",", ":"))
            )
        return ""


class TestAcceleratorDynamicLoader(GDBRemoteTestBase):
    def make_library(self):
        """Build the library object file the server will report."""
        path = self.getBuildArtifact("accelerator_lib.so")
        self.yaml2obj("accelerator_lib.yaml", path)
        return path

    def make_container(self, outer_path):
        """Embed a library in a larger file, as llvm-objcopy would when adding
        it to a container. Returns the container path and the (offset, size) of
        the embedded object.

        The embedded object puts .text at a different address than the outer
        one, so the assertions fail if the slice is ignored and the container is
        parsed from offset 0."""
        embedded_path = self.getBuildArtifact("embedded_lib.so")
        self.yaml2obj("embedded_lib.yaml", embedded_path)
        with open(outer_path, "rb") as f:
            outer_bytes = f.read()
        with open(embedded_path, "rb") as f:
            embedded_bytes = f.read()
        # The container must itself be a valid object file.
        prefix = outer_bytes + b"\x00" * ((-len(outer_bytes)) % 0x1000)
        container_path = self.getBuildArtifact("container.bin")
        with open(container_path, "wb") as f:
            f.write(prefix)
            f.write(embedded_bytes)
        return container_path, len(prefix), len(embedded_bytes)

    def find_module(self, target, path):
        basename = os.path.basename(path)
        for i in range(target.GetNumModules()):
            module = target.GetModuleAtIndex(i)
            if module.GetFileSpec().GetFilename() == basename:
                return module
        return None

    def connect_accelerator(self, library_infos):
        self.server.responder = AcceleratorResponder(library_infos)
        target = self.createTarget("accelerator.yaml")
        process = self.connect(target)
        self.assertTrue(process.IsValid(), "Process is valid")
        return target

    def assert_text_loaded_at(self, target, module, expected):
        section = module.FindSection(".text")
        self.assertTrue(section.IsValid(), "library should have a .text section")
        self.assertEqual(section.GetLoadAddress(target), expected)

    def test_whole_file_library(self):
        """A library given as a whole file is loaded at the reported address."""
        lib = self.make_library()
        target = self.connect_accelerator(
            [{"pathname": lib, "load": True, "load_address": 0x10000000}]
        )

        module = self.find_module(target, lib)
        self.assertIsNotNone(module, "library should be loaded into the target")
        # load_address slides the file, so .text (file address 0x1000) lands
        # 0x1000 past the base.
        self.assert_text_loaded_at(target, module, 0x10001000)

    def test_library_in_container(self):
        """A library embedded in a container file is located by offset/size."""
        lib = self.make_library()
        container, offset, size = self.make_container(lib)
        target = self.connect_accelerator(
            [
                {
                    "pathname": container,
                    "load": True,
                    "load_address": 0x20000000,
                    "file_offset": offset,
                    "file_size": size,
                }
            ]
        )

        module = self.find_module(target, container)
        self.assertIsNotNone(module, "embedded library should be loaded")
        # The embedded object has .text at 0x3000; the outer one has it at
        # 0x1000, so this only holds if the slice was used.
        self.assert_text_loaded_at(target, module, 0x20003000)

    def test_not_selected_for_host_target(self):
        """The loader is not used for a non-accelerator architecture."""
        self.server.responder = AcceleratorResponder([])
        target = self.createTarget("host.yaml")
        process = self.connect(target)
        self.assertTrue(process.IsValid(), "Process is valid")

        self.assertEqual(
            self.server.responder.dyld_queries,
            0,
            "a host target must not query the accelerator dynamic loader",
        )
