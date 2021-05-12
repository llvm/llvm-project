from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestHaltFails(GDBRemoteTestBase):

    class MyResponder(MockGDBServerResponder):
        def __init__(self, timeout):
            MockGDBServerResponder.__init__(self)
            self.timeout = timeout
            
        def setBreakpoint(self, packet):
            return "OK"
        
        def interrupt(self):
            # Simulate process waiting longer than the interrupt
            # timeout to stop, then sending the reply.
            time.sleep(self.timeout)
            return "T02reason:signal"
        
        def cont(self):
            # No response, wait for the client to interrupt us.
            return None
        
    def wait_for_and_check_event(self, wait_time, value):
        event = lldb.SBEvent()
        got_event = self.dbg.GetListener().WaitForEvent(wait_time, event)
        self.assertTrue(got_event, "Failed to get event after wait")
        self.assertTrue(lldb.SBProcess.EventIsProcessEvent(event), "Event was not a process event")
        event_type = lldb.SBProcess.GetStateFromEvent(event)
        self.assertEqual(event_type, value)
        
    def get_to_running(self, sleep_interval, interrupt_timeout = 5):
        self.server.responder = self.MyResponder(sleep_interval)
        self.target = self.createTarget("a.yaml")
        # Set a shorter (and known) interrupt timeout:
        self.dbg.HandleCommand("settings set target.process.interrupt-timeout {0}".format(interrupt_timeout))
        process = self.connect(self.target)
        self.dbg.SetAsync(True)

        # There should be a stopped event, consume that:
        self.wait_for_and_check_event(2, lldb.eStateStopped)
        process.Continue()

        # There should be a running event, consume that:
        self.wait_for_and_check_event(2, lldb.eStateRunning)
        return process

    @skipIfLinux # Failing on Linux, disabling to investigate
    @skipIfReproducer # FIXME: Unexpected packet during (passive) replay
    def test_destroy_while_running(self):
        process = self.get_to_running(10)
        process.Destroy()

        # Again pretend that after failing to be interrupted, we delivered the stop
        # and make sure we still exit properly.
        self.wait_for_and_check_event(14, lldb.eStateExited)
            
    @skipIfLinux # Failing on Linux, disabling to investigate
    @skipIfReproducer # FIXME: Unexpected packet during (passive) replay
    def test_async_interrupt(self):
        """
        Test that explicitly calling AsyncInterrupt, which then fails, leads
        to an "eStateExited" state.
        """
        process = self.get_to_running(10)
        # Now do the interrupt:
        process.SendAsyncInterrupt()

        # That should have caused the Halt to time out and we should
        # be in eStateExited:
        self.wait_for_and_check_event(15, lldb.eStateExited)

    @skipIfLinux # Failing on Linux, disabling to investigate
    @skipIfReproducer # FIXME: Unexpected packet during (passive) replay
    def test_interrupt_timeout(self):
        """
        Test that explicitly calling AsyncInterrupt but this time with
        an interrupt timeout longer than we're going to wait succeeds.
        """
        process = self.get_to_running(10, 20)
        # Now do the interrupt:
        process.SendAsyncInterrupt()

        # That should have caused the Halt to time out and we should
        # be in eStateExited:
        self.wait_for_and_check_event(20, lldb.eStateStopped)
        

        
