"""
Test the accelerator dynamic loader against a mock GDB server.

The loader is selected for accelerator architectures debugged over gdb-remote.
It then asks the server for the loaded libraries via
jAcceleratorPluginGetDynamicLoaderLibraryInfo and loads them into the target.
"""

import json
import os
import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

DYLD_PACKET = "jAcceleratorPluginGetDynamicLoaderLibraryInfo:"
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
BUNDLE_SECTION_OFFSET = 0x2000

CONTAINER_YAML = """\
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x1000
    Content:         "c3"
  - Name:            .hip_fatbin
    Type:            SHT_PROGBITS
    Offset:          0x2000
    AddressAlign:    0x1000
    Content:         "%s"
"""


def make_clang_offload_bundle(entries):
    """Return a legacy Clang offload bundle and its entry offsets."""
    ids = [entry_id.encode("utf-8") for entry_id, _ in entries]
    payload_offset = len(BUNDLE_MAGIC) + 8
    payload_offset += sum(24 + len(entry_id) for entry_id in ids)

    descriptors = bytearray()
    payloads = bytearray()
    offsets = []
    for entry_id, (_, payload) in zip(ids, entries):
        offsets.append(payload_offset)
        descriptors.extend(
            struct.pack("<QQQ", payload_offset, len(payload), len(entry_id))
        )
        descriptors.extend(entry_id)
        payloads.extend(payload)
        payload_offset += len(payload)

    return (
        BUNDLE_MAGIC + struct.pack("<Q", len(entries)) + descriptors + payloads,
        offsets,
    )


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

    def make_container(self):
        """Embed a library in a Clang offload bundle inside a host ELF.

        Returns the container path and the (offset, size) of the device object.

        The embedded object puts .text at a different address than the outer
        one, making it observable whether LLDB selected the bundled device
        image or the host ELF.
        """
        embedded_path = self.getBuildArtifact("embedded_lib.so")
        self.yaml2obj("embedded_lib.yaml", embedded_path)
        with open(embedded_path, "rb") as f:
            embedded_bytes = f.read()

        bundle, offsets = make_clang_offload_bundle(
            [
                ("host-x86_64-unknown-linux--", b""),
                ("hipv4-amdgcn-amd-amdhsa--gfx942", embedded_bytes),
            ]
        )
        yaml_path = self.getBuildArtifact("container.yaml")
        with open(yaml_path, "w") as f:
            f.write(CONTAINER_YAML % bundle.hex())
        container_path = self.getBuildArtifact("container.bin")
        self.yaml2obj(yaml_path, container_path)
        offset = BUNDLE_SECTION_OFFSET + offsets[1]
        with open(container_path, "rb") as f:
            container_bytes = f.read()
        self.assertEqual(
            container_bytes[offset : offset + len(embedded_bytes)], embedded_bytes
        )
        return container_path, offset, len(embedded_bytes)

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
        """A library embedded in a Clang bundle is loaded from the host ELF."""
        container, offset, size = self.make_container()
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
        # The bundled device object has .text at 0x3000; the host has it at
        # 0x1000.
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
