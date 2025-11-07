"""
Test that ConnectRemote sets ShouldDetach flag correctly.

When connecting to a remote process that stops after connection,
the process should be marked for detach (not kill) on destruction.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase
from lldbsuite.test import lldbutil


class TestConnectRemoteDetach(GDBRemoteTestBase):
    """Test that ConnectRemote properly sets ShouldDetach flag."""

    class StoppedResponder(MockGDBServerResponder):
        """A responder that returns a stopped process."""

        def qfThreadInfo(self):
            return "m1"

        def qsThreadInfo(self):
            return "l"

        def qC(self):
            return "QC1"

        def haltReason(self):
            # Return that we're stopped
            return "T05thread:1;"

        def cont(self):
            # Stay stopped
            return "T05thread:1;"

        def D(self):
            # Detach packet: this is what we want to verify gets called.
            return "OK"

        def k(self):
            # Kill packet: this is what we want to verify doesn't get called.
            raise RuntimeError("should not receive k(ill) packet")

    def test_connect_remote_sets_detach(self):
        """Test that ConnectRemote to a stopped process sets ShouldDetach."""
        self.server.responder = self.StoppedResponder()

        target = self.createTarget("a.yaml")
        process = self.connect(target)

        # Wait for the process to be in stopped state after connecting.
        # When ConnectRemote connects to a remote process that is stopped,
        # it should call SetShouldDetach(true) before CompleteAttach().
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # Now destroy the process. Because ShouldDetach was set to true
        # during ConnectRemote, this should send a 'D' (detach) packet
        # rather than a 'k' (kill) packet when the process is destroyed.
        process.Destroy()

        # Verify that the (D)etach packet was sent.
        self.assertPacketLogReceived(["D"])
