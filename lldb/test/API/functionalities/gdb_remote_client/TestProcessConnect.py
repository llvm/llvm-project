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
