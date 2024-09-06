import gdbremote_testcase
import random
import socket
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbgdbserverutils import Server
import lldbsuite.test.lldbplatformutil
from lldbgdbserverutils import Pipe


class TestGdbRemoteConnection(gdbremote_testcase.GdbRemoteTestCaseBase):
    @skipIfRemote  # reverse connect is not a supported use case for now
    def test_reverse_connect(self):
        # Reverse connect is the default connection method.
        self.connect_to_debug_monitor()
        # Verify we can do the handshake.  If that works, we'll call it good.
        self.do_handshake()

    @skipIfRemote
    def test_named_pipe(self):
        family, type, proto, _, addr = socket.getaddrinfo(
            self.stub_hostname, 0, proto=socket.IPPROTO_TCP
        )[0]
        self.sock = socket.socket(family, type, proto)
        self.sock.settimeout(self.DEFAULT_TIMEOUT)

        self.addTearDownHook(lambda: self.sock.close())

        pipe = Pipe(self.getBuildDir())

        self.addTearDownHook(lambda: pipe.close())

        args = self.debug_monitor_extra_args
        if lldb.remote_platform:
            args += ["*:0"]
        else:
            args += ["localhost:0"]

        args += ["--named-pipe", pipe.name]

        server = self.spawnSubprocess(
            self.debug_monitor_exe, args, install_remote=False
        )

        pipe.finish_connection(self.DEFAULT_TIMEOUT)
        port = pipe.read(10, self.DEFAULT_TIMEOUT)
        # Trim null byte, convert to int
        addr = (addr[0], int(port[:-1]))
        self.sock.connect(addr)
        self._server = Server(self.sock, server)

        # Verify we can do the handshake.  If that works, we'll call it good.
        self.do_handshake()
