from gdbclientutils import *

class TestGDBRemotePlatformFile(GDBPlatformClientTestBase):

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
                return "F-1,16"

        self.server.responder = Responder()

        self.match("platform file open /some/file.txt -v 0755",
                   [r"error: Invalid argument"],
                   error=True)
        self.match("platform file read 16 -o 11 -c 13",
                   [r"error: Invalid argument"],
                   error=True)
        self.match("platform file write 16 -o 11 -d teststring",
                   [r"error: Invalid argument"],
                   error=True)
        self.match("platform file close 16",
                   [r"error: Invalid argument"],
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
