from __future__ import print_function

import errno
import gdbremote_testcase
import lldbgdbserverutils
import re
import select
import socket
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStubReverseConnect(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Set up the test.
        gdbremote_testcase.GdbRemoteTestCaseBase.setUp(self)

        # Create a listener on a local port.
        self.listener_socket = self.create_listener_socket()
        self.assertIsNotNone(self.listener_socket)
        self.listener_port = self.listener_socket.getsockname()[1]

    def create_listener_socket(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except OSError as e:
            if e.errno != errno.EAFNOSUPPORT:
                raise
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        self.assertIsNotNone(sock)

        sock.settimeout(self.DEFAULT_TIMEOUT)
        if sock.family == socket.AF_INET:
            bind_addr = ("127.0.0.1", 0)
        elif sock.family == socket.AF_INET6:
            bind_addr = ("::1", 0)
        sock.bind(bind_addr)
        sock.listen(1)

        def tear_down_listener():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except:
                # ignore
                None

        self.addTearDownHook(tear_down_listener)
        return sock

    def reverse_connect_works(self):
        # Indicate stub startup should do a reverse connect.
        appended_stub_args = ["--reverse-connect"]
        if self.debug_monitor_extra_args:
            self.debug_monitor_extra_args += appended_stub_args
        else:
            self.debug_monitor_extra_args = appended_stub_args

        self.stub_hostname = "127.0.0.1"
        self.port = self.listener_port

        triple = self.dbg.GetSelectedPlatform().GetTriple()
        if re.match(".*-.*-.*-android", triple):
            self.forward_adb_port(
                self.port,
                self.port,
                "reverse",
                self.stub_device)

        # Start the stub.
        server = self.launch_debug_monitor(logfile=sys.stdout)
        self.assertIsNotNone(server)
        self.assertTrue(
            lldbgdbserverutils.process_is_running(
                server.pid, True))

        # Listen for the stub's connection to us.
        (stub_socket, address) = self.listener_socket.accept()
        self.assertIsNotNone(stub_socket)
        self.assertIsNotNone(address)
        print("connected to stub {} on {}".format(
            address, stub_socket.getsockname()))

        # Verify we can do the handshake.  If that works, we'll call it good.
        self.do_handshake(stub_socket)

        # Clean up.
        stub_socket.shutdown(socket.SHUT_RDWR)

    @debugserver_test
    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def test_reverse_connect_works_debugserver(self):
        self.init_debugserver_test(use_named_pipe=False)
        self.set_inferior_startup_launch()
        self.reverse_connect_works()

    @llgs_test
    @skipIfRemote  # reverse connect is not a supported use case for now
    def test_reverse_connect_works_llgs(self):
        self.init_llgs_test(use_named_pipe=False)
        self.set_inferior_startup_launch()
        self.reverse_connect_works()
