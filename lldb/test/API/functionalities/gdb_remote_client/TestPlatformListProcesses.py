import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestPlatformListProcesses(GDBRemoteTestBase):
    @skipIfRemote
    @skipIfWindows
    def test_get_all_processes(self):
        """Test listing processes"""

        class MyPlatformResponder(MockGDBServerResponder):
            def __init__(self):
                MockGDBServerResponder.__init__(self)
                self.done = False

            def qfProcessInfo(self, packet):
                return "pid:95117;name:666f6f;"

            def qsProcessInfo(self):
                if not self.done:
                    self.done = True
                    return "pid:95126;name:666f6f;"
                return "E10"

        self.server.responder = MyPlatformResponder()

        error = lldb.SBError()
        platform = lldb.SBPlatform("remote-linux")
        self.dbg.SetSelectedPlatform(platform)

        error = platform.ConnectRemote(
            lldb.SBPlatformConnectOptions(self.server.get_connect_url())
        )
        self.assertSuccess(error)
        self.assertTrue(platform.IsConnected())

        processes = platform.GetAllProcesses(error)
        self.assertSuccess(error)
        self.assertEqual(processes.GetSize(), 2)
        self.assertEqual(len(processes), 2)

        process_info = lldb.SBProcessInfo()
        processes.GetProcessInfoAtIndex(0, process_info)
        self.assertEqual(process_info.GetProcessID(), 95117)
        self.assertEqual(process_info.GetName(), "foo")

        processes.GetProcessInfoAtIndex(1, process_info)
        self.assertEqual(process_info.GetProcessID(), 95126)
        self.assertEqual(process_info.GetName(), "foo")

        platform.DisconnectRemote()
