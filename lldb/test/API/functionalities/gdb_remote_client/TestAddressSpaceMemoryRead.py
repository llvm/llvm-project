import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestAddressSpaceMemoryRead(GDBRemoteTestBase):
    """
    End-to-end test that the same numeric address read from two different
    address spaces returns different bytes. The mock server advertises two
    address spaces via "jAddressSpacesInfo" and answers "qMemRead" packets
    with different bytes depending on the requested address space.
    """

    def test(self):
        # JSON describing the two address spaces the mock process exposes. The
        # response goes over the gdb-remote channel, so the "}" characters need
        # to be escaped just like a real server would.
        address_spaces_json = (
            '[{"name":"global","value":1,"is_thread_specific":false},'
            '{"name":"local","value":2,"is_thread_specific":false}]'
        )

        class MyResponder(MockGDBServerResponder):
            def qSupported(self, client_supported):
                # Advertise address space support so the client will query it.
                return "PacketSize=3fff;QStartNoAckMode+;address-spaces+"

            def qHostInfo(self):
                return "ptrsize:8;endian:little;"

            def other(self, packet):
                if packet == "jAddressSpacesInfo":
                    return escape_binary(address_spaces_json)
                if packet.startswith("qMemRead:"):
                    # Parse the "key:value;" fields out of the packet body.
                    fields = {}
                    for field in packet[len("qMemRead:") :].split(";"):
                        if not field:
                            continue
                        key, _, value = field.partition(":")
                        fields[key] = value
                    space = int(fields["space"], 16)
                    if space == 1:
                        return "aabbccdd"
                    if space == 2:
                        return "11223344"
                    return "E01"
                return ""

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")
        process = self.connect(target)

        error = lldb.SBError()

        # Read the same numeric address (0x1000) from two different address
        # spaces and verify that the bytes differ and match the values the
        # server returned for each space.
        global_bytes = process.ReadMemoryFromSpec(
            lldb.SBAddressSpec(0x1000, "global"), 4, error
        )
        self.assertSuccess(error)
        self.assertEqual(global_bytes, b"\xaa\xbb\xcc\xdd")

        local_bytes = process.ReadMemoryFromSpec(
            lldb.SBAddressSpec(0x1000, "local"), 4, error
        )
        self.assertSuccess(error)
        self.assertEqual(local_bytes, b"\x11\x22\x33\x44")

        # Same address, different address space, different bytes.
        self.assertNotEqual(global_bytes, local_bytes)

        # The address space can also be given by its numeric id (2 == "local"),
        # which resolves to the same read as the "local" name.
        id_bytes = process.ReadMemoryFromSpec(lldb.SBAddressSpec(0x1000, 2), 4, error)
        self.assertSuccess(error)
        self.assertEqual(id_bytes, b"\x11\x22\x33\x44")
