from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbgdbclient import GDBPlatformClientTestBase

class TestGDBRemotePlatformFile(GDBPlatformClientTestBase):

    mydir = GDBPlatformClientTestBase.compute_mydir(__file__)

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
                return "F-1,58"

        self.server.responder = Responder()

        self.match("platform file open /some/file.txt -v 0755",
                   [r"File Descriptor = 16"])
        self.match("platform file read 16 -o 11 -c 13",
                   [r"Return = 11\nData = \"frobnicator\""])
        self.match("platform file write 16 -o 11 -d teststring",
                   [r"Return = 10"])
        self.match("platform file close 16",
                   [r"file 16 closed."])
        self.assertPacketLogContains([
            "vFile:open:2f736f6d652f66696c652e747874,00000202,000001ed",
            "vFile:pread:10,d,b",
            "vFile:pwrite:10,b,teststring",
            "vFile:close:10",
            ])

    def test_file_fail(self):
        """Test mocked failures of remote operations"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                # use ENOSYS as this constant differs between GDB Remote
                # Protocol and Linux, so we can test the translation
                return "F-1,58"

        self.server.responder = Responder()

        self.match("platform file open /some/file.txt -v 0755",
                   [r"error: Function not implemented"],
                   error=True)
        self.match("platform file read 16 -o 11 -c 13",
                   [r"error: Function not implemented"],
                   error=True)
        self.match("platform file write 16 -o 11 -d teststring",
                   [r"error: Function not implemented"],
                   error=True)
        self.match("platform file close 16",
                   [r"error: Function not implemented"],
                   error=True)
        self.assertPacketLogContains([
            "vFile:open:2f736f6d652f66696c652e747874,00000202,000001ed",
            "vFile:pread:10,d,b",
            "vFile:pwrite:10,b,teststring",
            "vFile:close:10",
            ])

    def test_file_size(self):
        """Test 'platform get-size'"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                return "F1000"

        self.server.responder = Responder()

        self.match("platform get-size /some/file.txt",
                   [r"File size of /some/file\.txt \(remote\): 4096"])
        self.assertPacketLogContains([
            "vFile:size:2f736f6d652f66696c652e747874",
            ])

    def test_file_size_fallback(self):
        """Test 'platform get-size fallback to vFile:fstat'"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                if packet.startswith("vFile:open:"):
                    return "F5"
                elif packet.startswith("vFile:fstat:"):
                    return "F40;" + 28 * "\0" + "\0\0\0\0\0\1\2\3" + 28 * "\0"
                if packet.startswith("vFile:close:"):
                    return "F0"
                return ""

        self.server.responder = Responder()

        self.match("platform get-size /some/file.txt",
                   [r"File size of /some/file\.txt \(remote\): 66051"])
        self.assertPacketLogContains([
            "vFile:size:2f736f6d652f66696c652e747874",
            "vFile:open:2f736f6d652f66696c652e747874,00000000,00000000",
            "vFile:fstat:5",
            "vFile:close:5",
            ])

        self.runCmd("platform disconnect")

        # For a new onnection, we should attempt vFile:size once again.
        server2 = MockGDBServer(self.server_socket_class())
        server2.responder = Responder()
        server2.start()
        self.addTearDownHook(lambda:server2.stop())
        self.runCmd("platform connect " + server2.get_connect_url())
        self.match("platform get-size /other/file.txt",
                   [r"File size of /other/file\.txt \(remote\): 66051"])

        self.assertPacketLogContains([
            "vFile:size:2f6f746865722f66696c652e747874",
            "vFile:open:2f6f746865722f66696c652e747874,00000000,00000000",
            "vFile:fstat:5",
            "vFile:close:5",
            ],
            log=server2.responder.packetLog)

    @skipIfWindows
    def test_file_permissions(self):
        """Test 'platform get-permissions'"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                return "F1a4"

        self.server.responder = Responder()

        self.match("platform get-permissions /some/file.txt",
                   [r"File permissions of /some/file\.txt \(remote\): 0o0644"])
        self.assertPacketLogContains([
            "vFile:mode:2f736f6d652f66696c652e747874",
            ])

    @skipIfWindows
    def test_file_permissions_fallback(self):
        """Test 'platform get-permissions' fallback to fstat"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                if packet.startswith("vFile:open:"):
                    return "F5"
                elif packet.startswith("vFile:fstat:"):
                    return "F40;" + 8 * "\0" + "\0\0\1\xA4" + 52 * "\0"
                if packet.startswith("vFile:close:"):
                    return "F0"
                return ""

        self.server.responder = Responder()

        try:
            self.match("platform get-permissions /some/file.txt",
                       [r"File permissions of /some/file\.txt \(remote\): 0o0644"])
            self.assertPacketLogContains([
                "vFile:mode:2f736f6d652f66696c652e747874",
                "vFile:open:2f736f6d652f66696c652e747874,00000000,00000000",
                "vFile:fstat:5",
                "vFile:close:5",
                ])
        finally:
            self.dbg.GetSelectedPlatform().DisconnectRemote()

    def test_file_exists(self):
        """Test 'platform file-exists'"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                return "F,1"

        self.server.responder = Responder()

        self.match("platform file-exists /some/file.txt",
                   [r"File /some/file\.txt \(remote\) exists"])
        self.assertPacketLogContains([
            "vFile:exists:2f736f6d652f66696c652e747874",
            ])

    def test_file_exists_not(self):
        """Test 'platform file-exists' with non-existing file"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                return "F,0"

        self.server.responder = Responder()

        self.match("platform file-exists /some/file.txt",
                   [r"File /some/file\.txt \(remote\) does not exist"])
        self.assertPacketLogContains([
            "vFile:exists:2f736f6d652f66696c652e747874",
            ])

    def test_file_exists_fallback(self):
        """Test 'platform file-exists' fallback to open"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                if packet.startswith("vFile:open:"):
                    return "F5"
                if packet.startswith("vFile:close:"):
                    return "F0"
                return ""

        self.server.responder = Responder()

        self.match("platform file-exists /some/file.txt",
                   [r"File /some/file\.txt \(remote\) exists"])
        self.assertPacketLogContains([
            "vFile:exists:2f736f6d652f66696c652e747874",
            "vFile:open:2f736f6d652f66696c652e747874,00000000,00000000",
            "vFile:close:5",
            ])

    def test_file_exists_not_fallback(self):
        """Test 'platform file-exists' fallback to open with non-existing file"""

        class Responder(MockGDBServerResponder):
            def vFile(self, packet):
                if packet.startswith("vFile:open:"):
                    return "F-1,2"
                return ""

        self.server.responder = Responder()

        self.match("platform file-exists /some/file.txt",
                   [r"File /some/file\.txt \(remote\) does not exist"])
        self.assertPacketLogContains([
            "vFile:exists:2f736f6d652f66696c652e747874",
            "vFile:open:2f736f6d652f66696c652e747874,00000000,00000000",
            ])
