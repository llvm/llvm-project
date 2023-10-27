"""
Test that stepping works even when the OS Plugin doesn't report
all threads at every stop.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestOSPluginStepping(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    @skipIf(oslist=["freebsd"], bugnumber="llvm.org/pr48352")
    def test_python_os_plugin(self):
        """Test that stepping works when the OS Plugin doesn't report all
        threads at every stop"""
        self.build()
        self.main_file = lldb.SBFileSpec("main.cpp")
        self.run_python_os_step_missing_thread(False)

    @skipIfWindows
    @skipIf(oslist=["freebsd"], bugnumber="llvm.org/pr48352")
    def test_python_os_plugin_prune(self):
        """Test that pruning the unreported PlanStacks works"""
        self.build()
        self.main_file = lldb.SBFileSpec("main.cpp")
        self.run_python_os_step_missing_thread(True)

    def get_os_thread(self):
        return self.process.GetThreadByID(0x111111111)

    def is_os_thread(self, thread):
        id = thread.GetID()
        return id == 0x111111111

    def run_python_os_step_missing_thread(self, do_prune):
        """Test that the Python operating system plugin works correctly"""

        # Our OS plugin does NOT report all threads:
        result = self.dbg.HandleCommand(
            "settings set process.experimental.os-plugin-reports-all-threads false"
        )

        python_os_plugin_path = os.path.join(self.getSourceDir(), "operating_system.py")
        (target, self.process, thread, thread_bkpt) = lldbutil.run_to_source_breakpoint(
            self, "first stop in thread - do a step out", self.main_file
        )

        main_bkpt = target.BreakpointCreateBySourceRegex(
            "Stop here and do not make a memory thread for thread_1", self.main_file
        )
        self.assertEqual(
            main_bkpt.GetNumLocations(), 1, "Main breakpoint has one location"
        )

        # There should not be an os thread before we load the plugin:
        self.assertFalse(
            self.get_os_thread().IsValid(), "No OS thread before loading plugin"
        )

        # Now load the python OS plug-in which should update the thread list and we should have
        # an OS plug-in thread overlaying thread_1 with id 0x111111111
        command = (
            "settings set target.process.python-os-plugin-path '%s'"
            % python_os_plugin_path
        )
        self.dbg.HandleCommand(command)

        # Verify our OS plug-in threads showed up
        os_thread = self.get_os_thread()
        self.assertTrue(
            os_thread.IsValid(),
            "Make sure we added the thread 0x111111111 after we load the python OS plug-in",
        )

        # Now we are going to step-out.  This should get interrupted by main_bkpt.  We've
        # set up the OS plugin so at this stop, we have lost the OS thread 0x111111111.
        # Make sure both of these are true:
        os_thread.StepOut()

        stopped_threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, main_bkpt
        )
        self.assertEqual(len(stopped_threads), 1, "Stopped at main_bkpt")
        thread = self.process.GetThreadByID(0x111111111)
        self.assertFalse(thread.IsValid(), "No thread 0x111111111 on second stop.")

        # Make sure we still have the thread plans for this thread:
        # First, don't show unreported threads, that should fail:
        command = "thread plan list -t 0x111111111"
        result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()
        interp.HandleCommand(command, result)
        self.assertFalse(
            result.Succeeded(), "We found no plans for the unreported thread."
        )
        # Now do it again but with the -u flag:
        command = "thread plan list -u -t 0x111111111"
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand(command, result)
        self.assertTrue(result.Succeeded(), "We found plans for the unreported thread.")

        if do_prune:
            # Prune the thread plan and continue, and we will run to exit.
            interp.HandleCommand("thread plan prune 0x111111111", result)
            self.assertTrue(
                result.Succeeded(), "Found the plan for 0x111111111 and pruned it"
            )

            # List again, make sure it doesn't work:
            command = "thread plan list -u -t 0x111111111"
            interp.HandleCommand(command, result)
            self.assertFalse(
                result.Succeeded(), "We still found plans for the unreported thread."
            )

            self.process.Continue()
            self.assertState(self.process.GetState(), lldb.eStateExited, "We exited.")
        else:
            # Now we are going to continue, and when we hit the step-out breakpoint, we will
            # put the OS plugin thread back, lldb will recover its ThreadPlanStack, and
            # we will stop with a "step-out" reason.
            self.process.Continue()
            os_thread = self.get_os_thread()
            self.assertTrue(os_thread.IsValid(), "The OS thread is back after continue")
            self.assertIn(
                "step out", os_thread.GetStopDescription(100), "Completed step out plan"
            )
