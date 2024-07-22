"""
Test lldb's handling of job control signals (SIGTSTP, SIGCONT).
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class JobControlTest(PExpectTest):
    @skipIf(macos_version=["<", "14.0"], asan=True)
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_job_control(self):
        def post_spawn():
            self.child.expect("PID=([0-9]+)")
            self.lldb_pid = int(self.child.match[1])

        run_under = [sys.executable, self.getSourcePath("shell.py")]
        self.launch(run_under=run_under, post_spawn=post_spawn)

        os.kill(self.lldb_pid, signal.SIGTSTP)
        self.child.expect("STATUS=([0-9]+)")
        status = int(self.child.match[1])

        self.assertTrue(os.WIFSTOPPED(status))
        self.assertEqual(os.WSTOPSIG(status), signal.SIGTSTP)

        os.kill(self.lldb_pid, signal.SIGCONT)

        self.child.sendline("quit")
        self.child.expect("RETURNCODE=0")
