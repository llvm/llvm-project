"""Test that we get thread names when interrupting a process."""


import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInterruptThreadNames(TestBase):
    @skipUnlessDarwin
    @add_test_categories(["pyapi"])
    def test_with_python_api(self):
        """Test that we get thread names when interrupting a process."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        launch_info = target.GetLaunchInfo()
        error = lldb.SBError()
        self.dbg.SetAsync(True)
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        listener = self.dbg.GetListener()
        broadcaster = process.GetBroadcaster()
        rc = broadcaster.AddListener(listener, lldb.SBProcess.eBroadcastBitStateChanged)
        self.assertNotEqual(rc, 0, "Unable to add listener to process")
        self.assertTrue(
            self.wait_for_running(process, listener),
            "Check that process is up and running",
        )

        inferior_set_up = self.wait_until_program_setup_complete(process, listener)

        # Check that the program was able to create its threads within the allotted time
        self.assertTrue(inferior_set_up.IsValid())
        self.assertEqual(inferior_set_up.GetValueAsSigned(), 1)

        self.check_number_of_threads(process)

        main_thread = lldb.SBThread()
        second_thread = lldb.SBThread()
        third_thread = lldb.SBThread()
        for idx in range(0, process.GetNumThreads()):
            t = process.GetThreadAtIndex(idx)
            if t.GetName() == "main thread":
                main_thread = t
            if t.GetName() == "second thread":
                second_thread = t
            if t.GetName() == "third thread":
                third_thread = t

        self.check_expected_threads_present(main_thread, second_thread, third_thread)

        process.Kill()

    # The process will set a global variable 'threads_up_and_running' to 1 when
    # it has has completed its setup.  Sleep for one second, pause the program,
    # check to see if the global has that value, and continue if it does not.
    def wait_until_program_setup_complete(self, process, listener):
        inferior_set_up = lldb.SBValue()
        retry = 5
        while retry > 0:
            arch = self.getArchitecture()
            # when running the testsuite against a remote arm device, it may take
            # a little longer for the process to start up.  Use a "can't possibly take
            # longer than this" value.
            if arch == "arm64" or arch == "armv7":
                time.sleep(10)
            else:
                time.sleep(1)
            process.SendAsyncInterrupt()
            self.assertTrue(
                self.wait_for_stop(process, listener), "Check that process is paused"
            )
            inferior_set_up = process.GetTarget().CreateValueFromExpression(
                "threads_up_and_running", "threads_up_and_running"
            )
            if inferior_set_up.IsValid() and inferior_set_up.GetValueAsSigned() == 1:
                retry = 0
            else:
                process.Continue()
            retry = retry - 1
        return inferior_set_up

    # Listen to the process events until we get an event saying that the process is
    # running.  Retry up to five times in case we get other events that are not
    # what we're looking for.
    def wait_for_running(self, process, listener):
        retry_count = 5
        if process.GetState() == lldb.eStateRunning:
            return True

        while retry_count > 0:
            event = lldb.SBEvent()
            listener.WaitForEvent(2, event)
            if event.GetType() == lldb.SBProcess.eBroadcastBitStateChanged:
                if process.GetState() == lldb.eStateRunning:
                    return True
            retry_count = retry_count - 1

        return False

    # Listen to the process events until we get an event saying the process is
    # stopped.  Retry up to five times in case we get other events that we are
    # not looking for.
    def wait_for_stop(self, process, listener):
        retry_count = 5
        if (
            process.GetState() == lldb.eStateStopped
            or process.GetState() == lldb.eStateCrashed
            or process.GetState() == lldb.eStateDetached
            or process.GetState() == lldb.eStateExited
        ):
            return True

        while retry_count > 0:
            event = lldb.SBEvent()
            listener.WaitForEvent(2, event)
            if event.GetType() == lldb.SBProcess.eBroadcastBitStateChanged:
                if (
                    process.GetState() == lldb.eStateStopped
                    or process.GetState() == lldb.eStateCrashed
                    or process.GetState() == lldb.eStateDetached
                    or process.GetState() == lldb.eStateExited
                ):
                    return True
                if (
                    process.GetState() == lldb.eStateCrashed
                    or process.GetState() == lldb.eStateDetached
                    or process.GetState() == lldb.eStateExited
                ):
                    return False
            retry_count = retry_count - 1

        return False

    def check_number_of_threads(self, process):
        self.assertEqual(
            process.GetNumThreads(),
            3,
            "Check that the process has three threads when sitting at the stopper() breakpoint",
        )

    def check_expected_threads_present(self, main_thread, second_thread, third_thread):
        self.assertTrue(
            main_thread.IsValid()
            and second_thread.IsValid()
            and third_thread.IsValid(),
            "Got all three expected threads",
        )
