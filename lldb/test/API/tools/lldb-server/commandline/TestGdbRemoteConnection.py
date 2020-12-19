from __future__ import print_function

import gdbremote_testcase
import select
import socket
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestGdbRemoteConnection(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote  # reverse connect is not a supported use case for now
    def test_reverse_connect_llgs(self):
        # Reverse connect is the default connection method.
        self.connect_to_debug_monitor()
        # Verify we can do the handshake.  If that works, we'll call it good.
        self.do_handshake(self.sock)

    @skipIfRemote
    @skipIfWindows
    def test_named_pipe_llgs(self):
        family, type, proto, _, addr = socket.getaddrinfo(
            self.stub_hostname, 0, proto=socket.IPPROTO_TCP)[0]
        self.sock = socket.socket(family, type, proto)
        self.sock.settimeout(self.DEFAULT_TIMEOUT)

        self.addTearDownHook(lambda: self.sock.close())

        named_pipe_path = self.getBuildArtifact("stub_port_number")

        # Create the named pipe.
        os.mkfifo(named_pipe_path)

        # Open the read side of the pipe in non-blocking mode.  This will
        # return right away, ready or not.
        named_pipe_fd = os.open(named_pipe_path, os.O_RDONLY | os.O_NONBLOCK)

        self.addTearDownHook(lambda: os.close(named_pipe_fd))

        args = self.debug_monitor_extra_args
        if lldb.remote_platform:
            args += ["*:0"]
        else:
            args += ["localhost:0"]

        args += ["--named-pipe", named_pipe_path]

        server = self.spawnSubprocess(
            self.debug_monitor_exe,
            args,
            install_remote=False)

        (ready_readers, _, _) = select.select(
            [named_pipe_fd], [], [], self.DEFAULT_TIMEOUT)
        self.assertIsNotNone(
            ready_readers,
            "write side of pipe has not written anything - stub isn't writing to pipe.")
        port = os.read(named_pipe_fd, 10)
        # Trim null byte, convert to int
        addr = (addr[0], int(port[:-1]))
        self.sock.connect(addr)

        # Verify we can do the handshake.  If that works, we'll call it good.
        self.do_handshake(self.sock)
