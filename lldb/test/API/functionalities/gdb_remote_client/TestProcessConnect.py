import lldb
import binascii
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


@skipIfRemote
class TestProcessConnect(GDBRemoteTestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    def test_gdb_remote_sync(self):
        """Test the gdb-remote command in synchronous mode"""
        try:
            self.dbg.SetAsync(False)
            self.expect("gdb-remote " + self.server.get_connect_address(),
                        substrs=['Process', 'stopped'])
        finally:
            self.dbg.GetSelectedPlatform().DisconnectRemote()

    @skipIfWindows
    @skipIfReproducer # Reproducer don't support async.
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
            self.dbg.GetSelectedPlatform().DisconnectRemote()

    @skipIfWindows
    @expectedFailureAll(oslist=["freebsd"])
    def test_process_connect_sync(self):
        """Test the gdb-remote command in synchronous mode"""
        try:
            self.dbg.SetAsync(False)
            self.expect("process connect connect://" +
                        self.server.get_connect_address(),
                        substrs=['Process', 'stopped'])
        finally:
            self.dbg.GetSelectedPlatform().DisconnectRemote()

    @skipIfWindows
    @expectedFailureAll(oslist=["freebsd"])
    @skipIfReproducer # Reproducer don't support async.
    def test_process_connect_async(self):
        """Test the gdb-remote command in asynchronous mode"""
        try:
            self.dbg.SetAsync(True)
            self.expect("process connect connect://" +
                        self.server.get_connect_address(),
                        matching=False,
                        substrs=['Process', 'stopped'])
            lldbutil.expect_state_changes(self, self.dbg.GetListener(),
                                          self.process(), [lldb.eStateStopped])
        finally:
            self.dbg.GetSelectedPlatform().DisconnectRemote()
