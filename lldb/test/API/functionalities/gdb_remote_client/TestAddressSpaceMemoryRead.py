import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestAddressSpaceMemoryRead(GDBRemoteTestBase):
    """
    End-to-end test that the same numeric address read from two different
    address spaces returns different bytes. The server advertises
    "address-spaces+" in qSupported, reports the spaces via "jAddressSpacesInfo",
    and reads memory with an optional "address_space:<hex-id>;" suffix on the
    standard memory packet.
    """

    def test(self):
        address_spaces_json = (
            '[{"name":"global","space_id":1,"is_thread_specific":false},'
            '{"name":"local","space_id":26,"is_thread_specific":true}]'
        )

        class MyResponder(MockGDBServerResponder):
            def qSupported(self, client_supported):
                return "PacketSize=3fff;QStartNoAckMode+;address-spaces+"

            def qHostInfo(self):
                return "ptrsize:8;endian:little;"

            def _bytes_for_space(self, space):
                if space == 1:
                    return "aabbccdd"
                if space == 26:
                    return "11223344"
                return "E01"

            def __init__(self):
                super().__init__()
                self.reads = []

            def _respond_impl(self, packet):
                # The base dispatcher can't parse the "address_space:<hex-id>;"
                # suffix, so handle suffixed reads here.
                if packet and packet[0] in ("m", "x") and "address_space:" in packet:
                    self.reads.append(packet)
                    space = 0
                    for field in packet[1:].split(";"):
                        key, _, value = field.partition(":")
                        if key == "address_space":
                            space = int(value, 16)
                    return self._bytes_for_space(space)
                return super()._respond_impl(packet)

            def x(self, addr, length):
                # Force the client onto the hex "m" read path.
                return ""

            def other(self, packet):
                if packet == "jAddressSpacesInfo":
                    return escape_binary(address_spaces_json)
                return ""

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")
        process = self.connect(target)

        error = lldb.SBError()

        # Same numeric address, two spaces (global == id 1, local == id 26).
        global_bytes = process.ReadMemory(lldb.SBProcessAddress(0x1000, 1), 4, error)
        self.assertSuccess(error)
        self.assertEqual(global_bytes, b"\xaa\xbb\xcc\xdd")

        # "local" is thread specific, so reading it needs a thread.
        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())
        local_bytes = process.ReadMemory(
            lldb.SBProcessAddress(0x1000, 26, thread), 4, error
        )
        self.assertSuccess(error)
        self.assertEqual(local_bytes, b"\x11\x22\x33\x44")

        # Same address, different address space, different bytes.
        self.assertNotEqual(global_bytes, local_bytes)

        # Ids go out in base 16, like the other numeric values in the packet.
        self.assertEqual(len(self.server.responder.reads), 2)
        self.assertIn("address_space:1;", self.server.responder.reads[0])
        self.assertIn("address_space:1a;", self.server.responder.reads[1])

        # Only the thread specific read carries a "thread:" field, and it names
        # the thread that was asked for.
        self.assertNotIn("thread:", self.server.responder.reads[0])
        self.assertIn(
            "thread:%x;" % thread.GetThreadID(), self.server.responder.reads[1]
        )

        # A thread specific space without a thread is an error, not a read.
        process.ReadMemory(lldb.SBProcessAddress(0x1000, 26), 4, error)
        self.assertTrue(error.Fail())
        self.assertIn("thread specific", error.GetCString())

        # Address spaces can be resolved by name, and reading through the
        # resolved id gives the same bytes as reading through the id directly.
        global_id = process.GetAddressSpaceID("global", error)
        self.assertSuccess(error)
        self.assertEqual(global_id, 1)
        self.assertEqual(
            process.ReadMemory(lldb.SBProcessAddress(0x1000, global_id), 4, error),
            global_bytes,
        )
        self.assertSuccess(error)

        self.assertEqual(process.GetAddressSpaceID("local", error), 26)
        self.assertSuccess(error)

        # An unknown name is an error rather than a silent default.
        process.GetAddressSpaceID("nonexistent", error)
        self.assertTrue(error.Fail())

        # So is no name at all.
        process.GetAddressSpaceID(None, error)
        self.assertTrue(error.Fail())
