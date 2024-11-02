import lldb
import binascii
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


@skipIfRemote
class TestProcessConnect(GDBRemoteTestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def test_gdb_remote_sync(self):
        """Test the gdb-remote command in synchronous mode"""
        try:
            self.dbg.SetAsync(False)
            self.expect("gdb-remote " + self.server.get_connect_address(),
                        substrs=['Process', 'stopped'])
        finally:
            self.dbg.GetSelectedTarget().GetProcess().Kill()

    def test_gdb_remote_async(self):
        """Test the gdb-remote command in asynchronous mode"""
        try:
            self.dbg.SetAsync(True)
            self.expect("gdb-remote " + self.server.get_connect_address(),
                        matching=False,
                        substrs=['Process', 'stopped'])
            lldbutil.expect_state_changes(self, self.dbg.GetListener(),
                                          self.process(), [lldb.eStateStopped])
        finally:
            self.dbg.GetSelectedTarget().GetProcess().Kill()
        lldbutil.expect_state_changes(self, self.dbg.GetListener(),
                                      self.process(), [lldb.eStateExited])

    @skipIfWindows
    def test_process_connect_sync(self):
        """Test the gdb-remote command in synchronous mode"""
        try:
            self.dbg.SetAsync(False)
            self.expect("platform select remote-gdb-server",
                        substrs=['Platform: remote-gdb-server', 'Connected: no'])
            self.expect("process connect " + self.server.get_connect_url(),
                        substrs=['Process', 'stopped'])
        finally:
            self.dbg.GetSelectedTarget().GetProcess().Kill()

    @skipIfWindows
    def test_process_connect_async(self):
        """Test the gdb-remote command in asynchronous mode"""
        try:
            self.dbg.SetAsync(True)
            self.expect("platform select remote-gdb-server",
                        substrs=['Platform: remote-gdb-server', 'Connected: no'])
            self.expect("process connect " + self.server.get_connect_url(),
                        matching=False,
                        substrs=['Process', 'stopped'])
            lldbutil.expect_state_changes(self, self.dbg.GetListener(),
                                          self.process(), [lldb.eStateStopped])
        finally:
            self.dbg.GetSelectedTarget().GetProcess().Kill()
        lldbutil.expect_state_changes(self, self.dbg.GetListener(),
                                      self.process(), [lldb.eStateExited])
    def test_breakpoint_count(self):
        """
        Test that breakpoint count gets reset for each new connection.
        """
        class MyResponder(MockGDBServerResponder):

            def __init__(self):
                super().__init__()
                self.continued = False

            def qfThreadInfo(self):
                return "m47"

            def qsThreadInfo(self):
                return "l"

            def setBreakpoint(self, packet):
                return "OK"

            def readRegister(self, reg):
                # Pretend we're at the breakpoint after we've been resumed.
                return "3412000000000000" if self.continued else "4747000000000000"

            def cont(self):
                self.continued = True
                return "T05thread=47;reason:breakpoint"

        # Connect to the first process and set our breakpoint.
        self.server.responder = MyResponder()
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        bkpt = target.BreakpointCreateByAddress(0x1234)
        self.assertTrue(bkpt.IsValid())
        self.assertEqual(bkpt.GetNumLocations(), 1)

        # "continue" the process. It should hit our breakpoint.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(bkpt.GetHitCount(), 1)

        # Now kill it. The breakpoint should still show a hit count of one.
        process.Kill()
        self.server.stop()
        self.assertEqual(bkpt.GetHitCount(), 1)

        # Start over, and reconnect.
        self.server = MockGDBServer(self.server_socket_class())
        self.server.start()

        process = self.connect(target)

        # The hit count should be reset.
        self.assertEqual(bkpt.GetHitCount(), 0)
