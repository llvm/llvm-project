import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


@skipIfWindows
class TestPty(GDBRemoteTestBase):
    mydir = TestBase.compute_mydir(__file__)
    server_socket_class = PtyServerSocket

    def test_process_connect_sync(self):
        """Test the process connect command in synchronous mode"""
        try:
            self.dbg.SetAsync(False)
            self.expect("platform select remote-gdb-server",
                        substrs=['Platform: remote-gdb-server', 'Connected: no'])
            self.expect("process connect " + self.server.get_connect_url(),
                        substrs=['Process', 'stopped'])
        finally:
            self.dbg.GetSelectedTarget().GetProcess().Kill()

    def test_process_connect_async(self):
        """Test the process connect command in asynchronous mode"""
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
