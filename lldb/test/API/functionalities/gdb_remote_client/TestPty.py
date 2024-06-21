import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


@skipIf(hostoslist=["windows"])
class TestPty(GDBRemoteTestBase):
    server_socket_class = PtyServerSocket

    def get_term_attrs(self):
        import termios

        return termios.tcgetattr(self._secondary_socket)

    def setUp(self):
        super().setUp()
        # Duplicate the pty descriptors so we can inspect the pty state after
        # they are closed
        self._primary_socket = os.dup(self.server._socket._primary.name)
        self._secondary_socket = os.dup(self.server._socket._secondary.name)
        self.orig_attr = self.get_term_attrs()

    def assert_raw_mode(self, current_attr):
        import termios

        self.assertEqual(
            current_attr[0]
            & (
                termios.BRKINT
                | termios.PARMRK
                | termios.ISTRIP
                | termios.INLCR
                | termios.IGNCR
                | termios.ICRNL
                | termios.IXON
            ),
            0,
        )
        self.assertEqual(current_attr[1] & termios.OPOST, 0)
        self.assertEqual(current_attr[2] & termios.CSIZE, termios.CS8)
        self.assertEqual(
            current_attr[3]
            & (termios.ICANON | termios.ECHO | termios.ISIG | termios.IEXTEN),
            0,
        )
        self.assertEqual(current_attr[6][termios.VMIN], 1)
        self.assertEqual(current_attr[6][termios.VTIME], 0)

    def get_parity_flags(self, attr):
        import termios

        return attr[2] & (termios.PARENB | termios.PARODD)

    def get_stop_bit_flags(self, attr):
        import termios

        return attr[2] & termios.CSTOPB

    def test_process_connect_sync(self):
        """Test the process connect command in synchronous mode"""
        try:
            self.dbg.SetAsync(False)
            self.expect(
                "platform select remote-gdb-server",
                substrs=["Platform: remote-gdb-server", "Connected: no"],
            )
            self.expect(
                "process connect " + self.server.get_connect_url(),
                substrs=["Process", "stopped"],
            )

            current_attr = self.get_term_attrs()
            # serial:// should set raw mode
            self.assert_raw_mode(current_attr)
            # other parameters should be unmodified
            self.assertEqual(current_attr[4:6], self.orig_attr[4:6])
            self.assertEqual(
                self.get_parity_flags(current_attr),
                self.get_parity_flags(self.orig_attr),
            )
            self.assertEqual(
                self.get_stop_bit_flags(current_attr),
                self.get_stop_bit_flags(self.orig_attr),
            )
        finally:
            self.dbg.GetSelectedTarget().GetProcess().Kill()
        # original mode should be restored on exit
        self.assertEqual(self.get_term_attrs(), self.orig_attr)

    def test_process_connect_async(self):
        """Test the process connect command in asynchronous mode"""
        try:
            self.dbg.SetAsync(True)
            self.expect(
                "platform select remote-gdb-server",
                substrs=["Platform: remote-gdb-server", "Connected: no"],
            )
            self.expect(
                "process connect " + self.server.get_connect_url(),
                matching=False,
                substrs=["Process", "stopped"],
            )
            lldbutil.expect_state_changes(
                self, self.dbg.GetListener(), self.process(), [lldb.eStateStopped]
            )

            current_attr = self.get_term_attrs()
            # serial:// should set raw mode
            self.assert_raw_mode(current_attr)
            # other parameters should be unmodified
            self.assertEqual(current_attr[4:6], self.orig_attr[4:6])
            self.assertEqual(
                self.get_parity_flags(current_attr),
                self.get_parity_flags(self.orig_attr),
            )
            self.assertEqual(
                self.get_stop_bit_flags(current_attr),
                self.get_stop_bit_flags(self.orig_attr),
            )
        finally:
            self.dbg.GetSelectedTarget().GetProcess().Kill()
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), self.process(), [lldb.eStateExited]
        )
        # original mode should be restored on exit
        self.assertEqual(self.get_term_attrs(), self.orig_attr)

    def test_connect_via_file(self):
        """Test connecting via the legacy file:// URL"""
        import termios

        try:
            self.expect(
                "platform select remote-gdb-server",
                substrs=["Platform: remote-gdb-server", "Connected: no"],
            )
            self.expect(
                "process connect file://" + self.server.get_connect_address(),
                substrs=["Process", "stopped"],
            )

            # file:// sets baud rate and some raw-related flags
            current_attr = self.get_term_attrs()
            self.assertEqual(
                current_attr[3]
                & (termios.ICANON | termios.ECHO | termios.ECHOE | termios.ISIG),
                0,
            )
            self.assertEqual(current_attr[4], termios.B115200)
            self.assertEqual(current_attr[5], termios.B115200)
            self.assertEqual(current_attr[6][termios.VMIN], 1)
            self.assertEqual(current_attr[6][termios.VTIME], 0)
        finally:
            self.dbg.GetSelectedTarget().GetProcess().Kill()

    def test_process_connect_params(self):
        """Test serial:// URL with parameters"""
        import termios

        try:
            self.expect(
                "platform select remote-gdb-server",
                substrs=["Platform: remote-gdb-server", "Connected: no"],
            )
            self.expect(
                "process connect "
                + self.server.get_connect_url()
                + "?baud=115200&stop-bits=2",
                substrs=["Process", "stopped"],
            )

            current_attr = self.get_term_attrs()
            self.assert_raw_mode(current_attr)
            self.assertEqual(current_attr[4:6], 2 * [termios.B115200])
            self.assertEqual(
                self.get_parity_flags(current_attr),
                self.get_parity_flags(self.orig_attr),
            )
            self.assertEqual(self.get_stop_bit_flags(current_attr), termios.CSTOPB)
        finally:
            self.dbg.GetSelectedTarget().GetProcess().Kill()
        # original mode should be restored on exit
        self.assertEqual(self.get_term_attrs(), self.orig_attr)
