import threading

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.gdbclientutils import MockGDBServerResponder
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestBreakpointUpdateInterrupt(GDBRemoteTestBase):
    class Responder(MockGDBServerResponder):
        def __init__(self):
            super().__init__()
            self.resumes = threading.Semaphore(0)
            self.packets = []

        def qfThreadInfo(self):
            return "m1"

        def qC(self):
            return "QC1"

        def haltReason(self):
            return "T05thread:1;"

        def cont(self):
            self.packets.append("continue")
            self.resumes.release()
            return self.RESPONSE_NONE

        def interrupt(self):
            self.packets.append("interrupt")
            return "T02thread:1;"

        def setBreakpoint(self, packet):
            self.packets.append(packet)
            return "OK"

        def clearBreakpoint(self, packet):
            self.packets.append(packet)
            return "OK"

    def check_operation(self, process, operation, action):
        responder = self.server.responder
        responder.packets.clear()
        result = operation()
        self.assertTrue(responder.resumes.acquire(timeout=5))
        packets = responder.packets
        self.assertEqual(packets[0], "interrupt")
        self.assertEqual(packets[-1], "continue")
        self.assertCountEqual(packets[1:-1], [f"{action}0,1000,1", f"{action}0,1001,1"])
        self.assertState(process.GetState(), lldb.eStateRunning)
        return result

    def test_multisite_breakpoint(self):
        """Check that packets for multiple sites share one inferior interruption."""

        responder = self.Responder()
        self.server.responder = responder
        self.runCmd("platform select remote-linux")
        target = self.createTarget("multisite-breakpoint.yaml")
        process = self.connect(target)
        listener = self.dbg.GetListener()
        lldbutil.expect_state_changes(self, listener, process, [lldb.eStateStopped])
        self.assertSuccess(target.SetModuleLoadAddress(target.GetModuleAtIndex(0), 0))
        self.dbg.SetAsync(True)
        self.assertSuccess(process.Continue())
        self.assertTrue(responder.resumes.acquire(timeout=5))
        self.assertEqual(responder.packets, ["continue"])
        lldbutil.expect_state_changes(self, listener, process, [lldb.eStateRunning])

        breakpoint = self.check_operation(
            process, lambda: target.BreakpointCreateByRegex("^site_"), "Z"
        )
        self.assertEqual(breakpoint.GetNumResolvedLocations(), 2)
        self.check_operation(process, lambda: breakpoint.SetEnabled(False), "z")
        self.assertEqual(breakpoint.GetNumResolvedLocations(), 0)
        self.check_operation(process, lambda: breakpoint.SetEnabled(True), "Z")
        self.assertEqual(breakpoint.GetNumResolvedLocations(), 2)
