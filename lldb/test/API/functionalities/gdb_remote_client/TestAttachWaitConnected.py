import socket
import threading
import time

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestAttachWaitConnected(GDBRemoteTestBase):
    TIMEOUT = 30

    @skipIfRemote
    @skipIfWindows
    # Flaky on Linux https://github.com/llvm/llvm-project/issues/236354.
    @skipIfLinux
    def test_interrupt_wait_for_attach(self):
        """
        An interrupt ends a wait-for attach on a process that was connected
        before the attach started.
        """
        waiting = threading.Event()

        class WaitingResponder(MockGDBServerResponder):
            def other(self, packet):
                if packet.startswith(("vAttachWait", "vAttachOrWait")):
                    waiting.set()
                    return self.RESPONSE_NONE
                return super().other(packet)

            def qC(self):
                # No process on the server yet, so the connection stays
                # "connected" instead of becoming an attached, stopped process.
                return ""

            def qfThreadInfo(self):
                # An empty reply would make the client assume a bare-metal
                # stub with pid 1.
                return "E01"

            def interrupt(self):
                return "E04"

        self.server.responder = WaitingResponder()
        target = self.dbg.CreateTarget("")
        process = self.connect(target)

        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        deadline = time.monotonic() + self.TIMEOUT
        while process.GetState() != lldb.eStateConnected:
            self.assertLess(
                time.monotonic(),
                deadline,
                "never became connected, state: %s pid: %s packets: %s"
                % (
                    lldb.SBDebugger.StateAsCString(process.GetState()),
                    process.GetProcessID(),
                    self.server.responder.packetLog.get_received(),
                ),
            )
            listener.WaitForEvent(1, event)

        attach_info = lldb.SBAttachInfo()
        attach_info.SetExecutable("no-such-process")
        attach_info.SetWaitForLaunch(True, False)

        attach_error = lldb.SBError()

        def attach():
            target.Attach(attach_info, attach_error)

        attach_thread = threading.Thread(target=attach, daemon=True)
        attach_thread.start()

        self.assertTrue(
            waiting.wait(self.TIMEOUT),
            "server never received the wait-for attach: %s" % attach_error.GetCString(),
        )
        process.SendAsyncInterrupt()
        attach_thread.join(self.TIMEOUT)
        if attach_thread.is_alive():
            # Teardown needs the target the attach is holding, so drop the
            # connection to let it finish before failing.
            self.server._socket._connection.shutdown(socket.SHUT_RDWR)
            attach_thread.join(self.TIMEOUT)
            self.fail("interrupt did not end the attach")
        self.assertTrue(attach_error.Fail())
