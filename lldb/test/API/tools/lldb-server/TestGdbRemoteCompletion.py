import tempfile
import gdbremote_testcase
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbgdbserverutils import *

class GdbRemoteCompletionTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):

    def init_lldb_server(self):
        self.debug_monitor_exe = get_lldb_server_exe()
        if not self.debug_monitor_exe:
            self.skipTest("lldb-server exe not found")
        port_file = tempfile.NamedTemporaryFile().name
        commandline_args = [
            "platform",
            "--listen",
            "*:0",
            "--socket-file",
            port_file
        ]
        server = self.spawnSubprocess(
            get_lldb_server_exe(),
            commandline_args,
            install_remote=False)
        self.assertIsNotNone(server)
        self.stub_hostname = "localhost"
        self.port = int(lldbutil.wait_for_file_on_target(self, port_file))
        self.sock = self.create_socket()
        self._server = Server(self.sock, server)

        self.do_handshake()

    def generate_hex_path(self, target):
        return str(os.path.join(self.getBuildDir(), target)).encode().hex()

    @add_test_categories(["llgs"])
    def test_autocomplete_path(self):
        self.build()
        self.init_lldb_server()

        # Test file-included completion when flag is set to 0.
        self.test_sequence.add_log_lines(
            ["read packet: $qPathComplete:0,{}#00".format(
                self.generate_hex_path("main")),
             "send packet: $M{},{}#00".format(
                self.generate_hex_path("main.d"),
                self.generate_hex_path("main.o"))
            ],
            True)

        # Test directory-only completion when flag is set to 1.
        os.makedirs(os.path.join(self.getBuildDir(), "test"))
        self.test_sequence.add_log_lines(
            ["read packet: $qPathComplete:1,{}#00".format(
                self.generate_hex_path("tes")),
             "send packet: $M{}{}#00".format(
                self.generate_hex_path("test"),
                os.path.sep.encode().hex()) # "test/" or "test\".
            ],
            True)

        self.expect_gdbremote_sequence()
