import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbgdbserverutils import get_debugserver_exe

import os
import platform
import shutil
import time
import socket


class TestStopAtEntry(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # The port used by debugserver.
    PORT = 54638

    # The number of attempts.
    ATTEMPTS = 10

    # Time given to the binary to launch and to debugserver to attach to it for
    # every attempt. We'll wait a maximum of 10 times 2 seconds while the
    # inferior will wait 10 times 10 seconds.
    TIMEOUT = 2

    def no_debugserver(self):
        if get_debugserver_exe() is None:
            return "no debugserver"
        return None

    def port_not_available(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if s.connect_ex(("127.0.0.1", self.PORT)) == 0:
            return "{} not available".format(self.PORT)
        return None

    @skipUnlessDarwin
    @skipIfRemote
    def test_stop_default_platform_sync(self):
        self.do_test_stop_at_entry(True, False)

    @skipUnlessDarwin
    @skipIfRemote
    def test_stop_default_platform_async(self):
        self.do_test_stop_at_entry(False, False)

    @skipUnlessDarwin
    @skipIfRemote
    @skipTestIfFn(no_debugserver)
    @skipTestIfFn(port_not_available)
    def test_stop_remote_platform_sync(self):
        self.do_test_stop_at_entry(True, True)

    @skipUnlessDarwin
    @skipIfRemote
    @skipTestIfFn(no_debugserver)
    @skipTestIfFn(port_not_available)
    def test_stop_remote_platform_async(self):
        self.do_test_stop_at_entry(False, True)

    def do_test_stop_at_entry(self, synchronous, remote):
        """Test the normal launch path in either sync or async mode"""
        self.build()

        target = lldbutil.run_to_breakpoint_make_target(self)
        launch_info = target.GetLaunchInfo()
        launch_info.SetLaunchFlags(lldb.eLaunchFlagStopAtEntry)
        old_async = self.dbg.GetAsync()

        def cleanup():
            self.dbg.SetAsync(old_async)

        self.addTearDownHook(cleanup)

        if not synchronous:
            self.dbg.SetAsync(True)
            listener = lldb.SBListener("test-process-listener")
            mask = listener.StartListeningForEventClass(
                self.dbg,
                lldb.SBProcess.GetBroadcasterClassName(),
                lldb.SBProcess.eBroadcastBitStateChanged,
            )
            self.assertEqual(
                mask,
                lldb.SBProcess.eBroadcastBitStateChanged,
                "Got right mask for listener",
            )
            launch_info.SetListener(listener)
        else:
            self.dbg.SetAsync(False)

        if remote:
            self.setup_remote_platform()

        error = lldb.SBError()

        process = target.Launch(launch_info, error)
        self.assertSuccess(error, "Launch failed")
        # If we are asynchronous, we have to wait for the events:
        if not synchronous:
            listener = launch_info.GetListener()
            event = lldb.SBEvent()
            result = listener.WaitForEvent(30, event)
            self.assertTrue(result, "Timed out waiting for event from process")
            state = lldb.SBProcess.GetStateFromEvent(event)
            self.assertState(
                state, lldb.eStateStopped, "Didn't get a stopped state after launch"
            )

        # Okay, we should be stopped.  Make sure we are indeed at the
        # entry point.  I only know how to do this on darwin:
        self.assertEqual(
            len(process.threads), 1, "Should only have one thread at entry"
        )
        thread = process.threads[0]
        frame = thread.GetFrameAtIndex(0)
        stop_func = frame.name
        self.assertEqual(stop_func, "_dyld_start")

        # Now make sure that we can resume the process and have it exit.
        error = process.Continue()
        self.assertSuccess(error, "Error continuing")
        # Fetch events till we get eStateExited:
        if not synchronous:
            # Get events till exited.
            listener = launch_info.GetListener()
            event = lldb.SBEvent()
            # We get two running events in a row here???  That's a bug
            # but not the one I'm testing for, so for now just fetch as
            # many as were sent.
            num_running = 0
            state = lldb.eStateRunning
            while state == lldb.eStateRunning:
                num_running += 1
                result = listener.WaitForEvent(30, event)
                self.assertTrue(result, "Timed out waiting for running")
                state = lldb.SBProcess.GetStateFromEvent(event)
                if num_running == 1:
                    self.assertState(state, lldb.eStateRunning, "Got running event")
            # The last event we should get is the exited event
            self.assertState(state, lldb.eStateExited, "Got exit event")
        else:
            # Make sure that the process has indeed exited
            state = process.GetState()
            self.assertState(state, lldb.eStateExited)

    def setup_remote_platform(self):
        return
        self.build()

        exe = self.getBuildArtifact("a.out")
        # Launch our test binary.

        # Attach to it with debugserver.
        debugserver = get_debugserver_exe()
        debugserver_args = ["localhost:{}".format(self.PORT)]
        self.spawnSubprocess(debugserver, debugserver_args)

        # Select the platform.
        self.expect("platform select remote-macosx", substrs=[sdk_dir])

        # Connect to debugserver
        interpreter = self.dbg.GetCommandInterpreter()
        connected = False
        for i in range(self.ATTEMPTS):
            result = lldb.SBCommandReturnObject()
            interpreter.HandleCommand("gdb-remote {}".format(self.PORT), result)
            connected = result.Succeeded()
            if connected:
                break
            time.sleep(self.TIMEOUT)

        self.assertTrue(connected, "could not connect to debugserver")
