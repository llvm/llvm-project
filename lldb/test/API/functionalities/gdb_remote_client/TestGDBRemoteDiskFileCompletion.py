from gdbclientutils import *

class TestGDBRemoteDiskFileCompletion(GDBRemoteTestBase):

    def test_autocomplete_request(self):
        """Test remote disk completion on remote-gdb-server plugin"""

        class Responder(MockGDBServerResponder):
            def qPathComplete(self):
                return "M{},{}".format(
                    "test".encode().hex(),
                    "123".encode().hex()
                )

        self.server.responder = Responder()

        try:
            self.runCmd("platform select remote-gdb-server")
            self.runCmd("platform connect connect://localhost:%d" %
                        self.server.port)
            self.assertTrue(self.dbg.GetSelectedPlatform().IsConnected())

            self.complete_from_to('platform get-size ', ['test', '123'])
            self.complete_from_to('platform get-file ', ['test', '123'])
            self.complete_from_to('platform put-file foo ', ['test', '123'])
            self.complete_from_to('platform file open ', ['test', '123'])
            self.complete_from_to('platform settings -w ', ['test', '123'])
        finally:
            self.dbg.GetSelectedPlatform().DisconnectRemote()
