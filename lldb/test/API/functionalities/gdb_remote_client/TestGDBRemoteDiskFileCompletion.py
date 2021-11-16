from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBPlatformClientTestBase

class TestGDBRemoteDiskFileCompletion(GDBPlatformClientTestBase):

    mydir = GDBPlatformClientTestBase.compute_mydir(__file__)

    def test_autocomplete_request(self):
        """Test remote disk completion on remote-gdb-server plugin"""

        class Responder(MockGDBServerResponder):
            def qPathComplete(self):
                return "M{},{}".format(
                    "test".encode().hex(),
                    "123".encode().hex()
                )

        self.server.responder = Responder()

        self.complete_from_to('platform get-size ', ['test', '123'])
        self.complete_from_to('platform get-file ', ['test', '123'])
        self.complete_from_to('platform put-file foo ', ['test', '123'])
        self.complete_from_to('platform file open ', ['test', '123'])
        self.complete_from_to('platform settings -w ', ['test', '123'])
