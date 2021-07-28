from gdbclientutils import *

class TestGDBRemotePlatformFile(GDBRemoteTestBase):

    def test_file_open(self):
        """Test mock-opening a remote file"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                return "F10"

        self.server.responder = Responder()

        try:
            self.runCmd("platform select remote-gdb-server")
            self.runCmd("platform connect connect://" +
                        self.server.get_connect_address())
            self.assertTrue(self.dbg.GetSelectedPlatform().IsConnected())

            self.runCmd("platform file open /some/file.txt -v 0755")
            self.assertPacketLogContains([
                "vFile:open:2f736f6d652f66696c652e747874,0000020a,000001ed"
                ])
        finally:
            self.dbg.GetSelectedPlatform().DisconnectRemote()
