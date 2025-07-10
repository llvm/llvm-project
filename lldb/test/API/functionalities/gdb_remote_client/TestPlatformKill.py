import lldb
import time
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestPlatformKill(GDBRemoteTestBase):
    @skipIfRemote
    def test_kill_different_platform(self):
        """Test connecting to a remote linux platform"""

        self.build(dictionary={"CXX_SOURCES": "sleep.cpp"})
        host_process = self.spawnSubprocess(self.getBuildArtifact())

        # Create a fake remote process with the same PID as host_process
        class MyResponder(MockGDBServerResponder):
            def __init__(self):
                MockGDBServerResponder.__init__(self)
                self.got_kill = False

            def qC(self):
                return "QC%x" % host_process.pid

            def k(self):
                self.got_kill = True
                return "X09"

        self.server.responder = MyResponder()

        error = lldb.SBError()
        target = self.dbg.CreateTarget(
            "", "x86_64-pc-linux", "remote-linux", False, error
        )
        self.assertSuccess(error)
        process = self.connect(target)
        self.assertEqual(process.GetProcessID(), host_process.pid)

        host_platform = lldb.SBPlatform("host")
        self.assertSuccess(host_platform.Kill(host_process.pid))

        # Host dies, remote process lives.
        self.assertFalse(self.server.responder.got_kill)
        self.assertIsNotNone(host_process.wait(timeout=10))

        # Now kill the remote one as well
        self.assertSuccess(process.Kill())
        self.assertTrue(self.server.responder.got_kill)
