"""
Test process attach when executable was deleted.
"""



import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestDeletedExecutable(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows # cannot delete a running executable
    @expectedFailureAll(oslist=["linux"],
        triple=no_match('aarch64-.*-android'))
        # determining the architecture of the process fails
    @expectedFailureNetBSD
    @skipIfReproducer # File synchronization is not supported during replay.
    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(self,
            "token_pid_%d" % (int(os.getpid())))
        self.addTearDownHook(
            lambda: self.run_platform_command(
                "rm %s" %
                (pid_file_path)))

        # Spawn a new process
        popen = self.spawnSubprocess(exe, [pid_file_path])
        self.addTearDownHook(self.cleanupSubprocesses)

        # Wait until process has fully started up.
        pid = lldbutil.wait_for_file_on_target(self, pid_file_path)

        # Now we can safely remove the executable and test if we can attach.
        os.remove(exe)

        self.runCmd("process attach -p " + str(popen.pid))
        self.runCmd("kill")
