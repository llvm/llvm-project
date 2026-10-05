"""
Test the accelerator dynamic loader against a mock GDB server.

The server selects the loader by name through jLLDBSettings. The loader
then asks the server for the loaded libraries via
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

    def __init__(
        self,
        library_infos,
        dyld_plugin_name="accelerator-gdb-remote",
    ):
        MockGDBServerResponder.__init__(self)
        self.library_infos = library_infos
        self.dyld_plugin_name = dyld_plugin_name
        self.dyld_queries = 0
        self.last_dyld_packet = None

    def qSupported(self, client_supported):
        features = super().qSupported(client_supported) + ";qXfer:features:read+"
        if self.dyld_plugin_name is not None:
            features += ";lldb-settings+"
        return features

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
        if packet == "jLLDBSettings":
            response = {
                "dyld_plugin_name": self.dyld_plugin_name,
                "gpu_plugin_name": "mock",
                "send_dyld_packet_to_gpu": True,
            }
            return escape_binary(json.dumps(response, separators=(",", ":")))
        if packet.startswith(DYLD_PACKET):
            self.dyld_queries += 1
            self.last_dyld_packet = packet
            # "}" is the gdb-remote escape character.
            return escape_binary(
                json.dumps({"library_infos": self.library_infos}, separators=(",", ":"))
            )
        return ""


class TestAcceleratorDynamicLoader(GDBRemoteTestBase):
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
        self.assertIsNotNone(self.server.responder.last_dyld_packet)
        self.assertIn('"full":true', self.server.responder.last_dyld_packet)
        return target

    def assert_text_loaded_at(self, target, module, expected):
        section = module.FindSection(".text")
        self.assertTrue(section.IsValid(), "library should have a .text section")
        self.assertEqual(section.GetLoadAddress(target), expected)

    def test_whole_file_library(self):
        """A library given as a whole file is loaded at the reported address."""
        lib = self.getBuildArtifact("accelerator_lib.so")
        self.yaml2obj("accelerator_lib.yaml", lib)
        target = self.connect_accelerator(
            [{"pathname": lib, "load": True, "load_address": 0x10000000}]
        )

        module = self.find_module(target, lib)
        self.assertIsNotNone(module, "library should be loaded into the target")
        # load_address slides the file, so .text (file address 0x1000) lands
        # 0x1000 past the base.
        self.assert_text_loaded_at(target, module, 0x10001000)

    def test_library_unload(self):
        """An unload clears section addresses and broadcasts the module."""
        lib = self.getBuildArtifact("accelerator_lib.so")
        self.yaml2obj("accelerator_lib.yaml", lib)
        target = self.createTarget("accelerator.yaml")
        module = target.AddModule(lib, None, None)
        self.assertTrue(module.IsValid(), "library should be added to the target")

        text = module.FindSection(".text")
        self.assertTrue(text.IsValid(), "library should have a .text section")
        error = target.SetModuleLoadAddress(module, 0x10000000)
        self.assertSuccess(error)
        self.assert_text_loaded_at(target, module, 0x10001000)

        listener = lldb.SBListener("accelerator-module-unload")
        listened = target.GetBroadcaster().AddListener(
            listener, lldb.SBTarget.eBroadcastBitModulesUnloaded
        )
        self.assertEqual(listened, lldb.SBTarget.eBroadcastBitModulesUnloaded)

        self.server.responder = AcceleratorResponder([{"pathname": lib, "load": False}])
        process = self.connect(target)
        self.assertTrue(process.IsValid(), "Process is valid")
        self.assertEqual(self.server.responder.dyld_queries, 1)
        self.assertEqual(text.GetLoadAddress(target), lldb.LLDB_INVALID_ADDRESS)

        event = lldb.SBEvent()
        self.assertTrue(listener.WaitForEvent(1, event), "got module unload event")
        self.assertEqual(event.GetType(), lldb.SBTarget.eBroadcastBitModulesUnloaded)
        self.assertEqual(lldb.SBTarget.GetNumModulesFromEvent(event), 1)
        unloaded = lldb.SBTarget.GetModuleAtIndexFromEvent(0, event)
        self.assertEqual(unloaded.GetFileSpec().GetFilename(), os.path.basename(lib))

    def test_accelerator_lldb_settings_not_supported(self):
        """The loader is not used unless the server selects it by name."""
        self.server.responder = AcceleratorResponder([], dyld_plugin_name=None)
        target = self.createTarget("accelerator.yaml")
        process = self.connect(target)
        self.assertTrue(process.IsValid(), "Process is valid")

        self.assertEqual(
            self.server.responder.dyld_queries,
            0,
            "the target architecture must not select the accelerator loader",
        )
        self.assertIsNone(self.server.responder.last_dyld_packet)
