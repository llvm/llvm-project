import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestGDBRemoteLoad(GDBRemoteTestBase):

    @expectedFailureAll(archs=["aarch64"], oslist=["freebsd"],
                        bugnumber="llvm.org/pr49414")
    def test_module_load_address(self):
        """Test that setting the load address of a module uses virtual addresses"""
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())
        self.assertTrue(target.SetModuleLoadAddress(module, 0).Success())
        address = target.ResolveLoadAddress(0x2001)
        self.assertTrue(address.IsValid())
        self.assertEqual(".data", address.GetSection().GetName())

    @skipIfReproducer # Packet log is not populated during replay.
    @expectedFailureAll(archs=["aarch64"], oslist=["freebsd"],
                        bugnumber="llvm.org/pr49414")
    def test_ram_load(self):
        """Test loading an object file to a target's ram"""
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        self.dbg.HandleCommand("target modules load -l -s0")
        self.assertPacketLogContains([
                "M1000,4:c3c3c3c3",
                "M1004,2:3232"
                ])

    @skipIfXmlSupportMissing
    @skipIfReproducer # Packet log is not populated during replay.
    @expectedFailureAll(archs=["aarch64"], oslist=["freebsd"],
                        bugnumber="llvm.org/pr49414")
    def test_flash_load(self):
        """Test loading an object file to a target's flash memory"""

        class Responder(MockGDBServerResponder):
            def qSupported(self, client_supported):
                return "PacketSize=3fff;QStartNoAckMode+;qXfer:memory-map:read+"

            def qXferRead(self, obj, annex, offset, length):
                if obj == "memory-map":
                    return (self.MEMORY_MAP[offset:offset + length],
                            offset + length < len(self.MEMORY_MAP))
                return None, False

            def other(self, packet):
                if packet[0:11] == "vFlashErase":
                    return "OK"
                if packet[0:11] == "vFlashWrite":
                    return "OK"
                if packet == "vFlashDone":
                    return "OK"
                return ""

            MEMORY_MAP = """<?xml version="1.0"?>
<memory-map>
  <memory type="ram" start="0x0" length="0x1000"/>
  <memory type="flash" start="0x1000" length="0x1000">
    <property name="blocksize">0x100</property>
  </memory>
  <memory type="ram" start="0x2000" length="0x1D400"/>
</memory-map>
"""

        self.server.responder = Responder()
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        self.dbg.HandleCommand("target modules load -l -s0")
        self.assertPacketLogContains([
                "vFlashErase:1000,100",
                "vFlashWrite:1000:\xc3\xc3\xc3\xc3",
                "vFlashWrite:1004:\x32\x32",
                "vFlashDone"
                ])
