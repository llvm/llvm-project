"""
Test that a process plugin keeps correctly resuming, detaching, and
reporting events for the real threads backing an OS plugin's virtual
threads, wherever the OS plugin replaces a real thread in the user-facing
thread list.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

# The tid the OS plugin in this directory reports for its virtual thread.
OS_TID = 0x111111111


@skipIfTargetDoesNotSupportThreads()
class TestOSPluginBackingThreadEvents(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.source = lldb.SBFileSpec("main.cpp")

    def stop_and_load_os_plugin(self, stop_regex, args=None):
        """Run to stop_regex, load the OS plugin, and return (target, process).

        Asserts the premise the plugin sets up: the real thread the process
        stopped on is gone from the user-facing list, replaced by a virtual
        thread that is not the list's first entry.
        """
        self.build()
        launch_info = None
        if args:
            launch_info = lldb.SBLaunchInfo(args)
            launch_info.SetWorkingDirectory(self.get_process_working_directory())
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, stop_regex, self.source, launch_info=launch_info
        )

        # main is core 0; the worker thread keeps a second real thread around.
        self.assertGreaterEqual(process.GetNumThreads(), 2)
        real_tid = thread.GetThreadID()
        self.assertEqual(process.GetThreadAtIndex(0).GetThreadID(), real_tid)

        self.runCmd(
            "settings set target.process.python-os-plugin-path '%s'"
            % os.path.join(self.getSourceDir(), "operating_system.py")
        )

        os_thread = process.GetThreadByID(OS_TID)
        self.assertTrue(os_thread.IsValid(), "the OS plugin thread showed up")
        self.assertFalse(
            process.GetThreadByID(real_tid).IsValid(),
            "the real thread we stopped on is no longer user-visible",
        )
        self.assertNotEqual(
            process.GetThreadAtIndex(0).GetThreadID(),
            OS_TID,
            "the virtual thread is not the first thread in the list",
        )
        return target, process

    def test_breakpoint_on_backed_thread(self):
        """A breakpoint hit on a real thread is reported on the virtual thread
        standing in for it, not on whichever thread happens to be first."""
        target, process = self.stop_and_load_os_plugin("// Break here")

        breakpoint = target.BreakpointCreateBySourceRegex(
            "// Second stop here", self.source
        )
        self.assertEqual(breakpoint.GetNumLocations(), 1)

        # Resuming has to walk the real threads: the virtual thread has no OS
        # thread to resume.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)

        stopped = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(len(stopped), 1, "exactly one thread hit the breakpoint")
        self.assertEqual(stopped[0].GetThreadID(), OS_TID)

    def test_watchpoint_on_backed_thread(self):
        """A watchpoint is programmed into the real threads' debug registers and
        its hit is reported on the virtual thread."""
        target, process = self.stop_and_load_os_plugin("// Break here")

        self.runCmd("watchpoint set variable g_watched")
        self.assertEqual(target.GetNumWatchpoints(), 1)

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonWatchpoint)
        self.assertIsNotNone(thread, "stopped for the watchpoint")
        self.assertEqual(thread.GetThreadID(), OS_TID)

        self.runCmd("watchpoint delete 1")
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)

    def test_detach_with_virtual_threads(self):
        """Detaching resumes the real threads rather than the virtual one, so the
        inferior runs on afterwards."""
        marker = self.getBuildArtifact("detached.marker")
        if os.path.exists(marker):
            os.remove(marker)

        _, process = self.stop_and_load_os_plugin("// Break here", args=[marker])

        self.assertFalse(os.path.exists(marker), "marker was not yet created")
        self.assertSuccess(process.Detach())
        self.assertState(process.GetState(), lldb.eStateDetached)

        # main writes the marker just before returning. A thread left suspended
        # by the detach never gets there.
        while not os.path.exists(marker):
            time.sleep(0.1)
        self.assertTrue(os.path.exists(marker), "the inferior ran to completion")
