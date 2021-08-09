from gdbclientutils import *

class TestGDBRemotePlatformFile(GDBRemoteTestBase):

    def test_file(self):
        """Test mock operations on a remote file"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                if packet.startswith("vFile:open:"):
                    return "F10"
                elif packet.startswith("vFile:pread:"):
                    return "Fd;frobnicator"
                elif packet.startswith("vFile:pwrite:"):
                    return "Fa"
                elif packet.startswith("vFile:close:"):
                    return "F0"
                return "F-1,16"

        self.server.responder = Responder()

        try:
            self.runCmd("platform select remote-gdb-server")
            self.runCmd("platform connect connect://" +
                        self.server.get_connect_address())
            self.assertTrue(self.dbg.GetSelectedPlatform().IsConnected())

            self.match("platform file open /some/file.txt -v 0755",
                       [r"File Descriptor = 16"])
            self.match("platform file read 16 -o 11 -c 13",
                       [r"Return = 11\nData = \"frobnicator\""])
            self.match("platform file write 16 -o 11 -d teststring",
                       [r"Return = 10"])
            self.match("platform file close 16",
                       [r"file 16 closed."])
            self.assertPacketLogContains([
                "vFile:open:2f736f6d652f66696c652e747874,0000020a,000001ed",
                "vFile:pread:10,d,b",
                "vFile:pwrite:10,b,teststring",
                "vFile:close:10",
                ])
        finally:
            self.dbg.GetSelectedPlatform().DisconnectRemote()

    def test_file_fail(self):
        """Test mocked failures of remote operations"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                return "F-1,16"

        self.server.responder = Responder()

        try:
            self.runCmd("platform select remote-gdb-server")
            self.runCmd("platform connect connect://" +
                        self.server.get_connect_address())
            self.assertTrue(self.dbg.GetSelectedPlatform().IsConnected())

            self.match("platform file open /some/file.txt -v 0755",
                       [r"error: Invalid argument"],
                       error=True)
            # TODO: fix the commands to fail on unsuccessful result
            self.match("platform file read 16 -o 11 -c 13",
                       [r"Return = -1\nData = \"\""])
            self.match("platform file write 16 -o 11 -d teststring",
                       [r"Return = -1"])
            self.match("platform file close 16",
                       [r"error: Invalid argument"],
                       error=True)
            self.assertPacketLogContains([
                "vFile:open:2f736f6d652f66696c652e747874,0000020a,000001ed",
                "vFile:pread:10,d,b",
                "vFile:pwrite:10,b,teststring",
                "vFile:close:10",
                ])
        finally:
            self.dbg.GetSelectedPlatform().DisconnectRemote()
