import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbgdbserverutils import *

import xml.etree.ElementTree as ET


@skipIfWindows
class PtyServerTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):

    def setUp(self):
        super().setUp()
        import pty
        import tty
        primary, secondary = pty.openpty()
        tty.setraw(primary)
        self._primary = io.FileIO(primary, 'r+b')
        self._secondary = io.FileIO(secondary, 'r+b')

    def get_debug_monitor_command_line_args(self, attach_pid=None):
        commandline_args = self.debug_monitor_extra_args
        if attach_pid:
            commandline_args += ["--attach=%d" % attach_pid]

        libc = ctypes.CDLL(None)
        libc.ptsname.argtypes = (ctypes.c_int,)
        libc.ptsname.restype = ctypes.c_char_p
        pty_path = libc.ptsname(self._primary.fileno()).decode()
        commandline_args += ["serial://%s" % (pty_path,)]
        return commandline_args

    def connect_to_debug_monitor(self, attach_pid=None):
        self.reverse_connect = False
        server = self.launch_debug_monitor(attach_pid=attach_pid)
        self.assertIsNotNone(server)

        # TODO: make it into proper abstraction
        class FakeSocket:
            def __init__(self, fd):
                self.fd = fd

            def sendall(self, frame):
                self.fd.write(frame)

            def recv(self, count):
                return self.fd.read(count)

        self.sock = FakeSocket(self._primary)
        self._server = Server(self.sock, server)
        return server

    @add_test_categories(["llgs"])
    def test_pty_server(self):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior()

        # target.xml transfer should trigger a large enough packet to check
        # for partial write regression
        self.test_sequence.add_log_lines([
            "read packet: $qXfer:features:read:target.xml:0,200000#00",
            {
                "direction": "send",
                "regex": re.compile("^\$l(.+)#[0-9a-fA-F]{2}$", flags=re.DOTALL),
                "capture": {1: "target_xml"},
            }],
            True)
        context = self.expect_gdbremote_sequence()
        # verify that we have received a complete, non-malformed XML
        self.assertIsNotNone(ET.fromstring(context.get("target_xml")))
