"""
Test that the process continues running after we detach from it.
"""

import lldb
import time
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DetachResumesTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_detach_resumes(self):
        self.build()
        exe = self.getBuildArtifact()

        # The inferior will use this file to let us know it is ready to be
        # attached.
        sync_file_path = lldbutil.append_to_process_working_directory(
            self, "sync_file_%d" % (int(time.time()))
        )

        # And this one to let us know it is running after we've detached from
        # it.
        exit_file_path = lldbutil.append_to_process_working_directory(
            self, "exit_file_%d" % (int(time.time()))
        )

        popen = self.spawnSubprocess(
            self.getBuildArtifact(exe), [sync_file_path, exit_file_path]
        )
        lldbutil.wait_for_file_on_target(self, sync_file_path)

        self.runCmd("process attach -p " + str(popen.pid))

        # Set a breakpoint at a place that will be called by multiple threads
        # simultaneously. On systems (e.g. linux) where the debugger needs to
        # send signals to suspend threads, these signals will race with threads
        # hitting the breakpoint (and stopping on their own).
        bpno = lldbutil.run_break_set_by_symbol(self, "break_here")

        # And let the inferior know it can call the function.
        self.runCmd("expr -- wait_for_debugger_flag = false")

        self.runCmd("continue")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Detach, the process should keep running after this, and not be stopped
        # by the signals that the debugger may have used to suspend the threads.
        self.runCmd("detach")

        lldbutil.wait_for_file_on_target(self, exit_file_path)
