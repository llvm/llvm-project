from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestRestartBug(GDBRemoteTestBase):

    @expectedFailureAll(bugnumber="llvm.org/pr24530")
    def test(self):
        """
        Test auto-continue behavior when a process is interrupted to deliver
        an "asynchronous" packet. This simulates the situation when a process
        stops on its own just as lldb client is about to interrupt it. The
        client should not auto-continue in this case, unless the user has
        explicitly requested that we ignore signals of this type.
        """
        class MyResponder(MockGDBServerResponder):
            continueCount = 0

            def setBreakpoint(self, packet):
                return "OK"

            def interrupt(self):
                # Simulate process stopping due to a raise(SIGINT) just as lldb
                # is about to interrupt it.
                return "T02reason:signal"

            def cont(self):
                self.continueCount += 1
                if self.continueCount == 1:
                    # No response, wait for the client to interrupt us.
                    return None
                return "W00" # Exit

        self.server.responder = MyResponder()
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        self.dbg.SetAsync(True)
        process.Continue()

        # resume the process and immediately try to set another breakpoint. When using the remote
        # stub, this will trigger a request to stop the process.  Make sure we
        # do not lose this signal.
        bkpt = target.BreakpointCreateByAddress(0x1234)
        self.assertTrue(bkpt.IsValid())
        self.assertEqual(bkpt.GetNumLocations(), 1)

        event = lldb.SBEvent()
        while self.dbg.GetListener().WaitForEvent(2, event):
            if self.TraceOn():
                print("Process changing state to:",
                    self.dbg.StateAsCString(process.GetStateFromEvent(event)))
            if process.GetStateFromEvent(event) == lldb.eStateExited:
                break

        # We should get only one continue packet as the client should not
        # auto-continue after setting the breakpoint.
        self.assertEqual(self.server.responder.continueCount, 1)
        # And the process should end up in the stopped state.
        self.assertEqual(process.GetState(), lldb.eStateStopped)
