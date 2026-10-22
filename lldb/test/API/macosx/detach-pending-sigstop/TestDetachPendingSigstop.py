"""
Test that a process keeps running after lldb attaches to it and detaches,
with and without an expression evaluated in between.

To detach, debugserver stops the process with SIGSTOP. If PT_DETACH runs
before the stopped thread has consumed the reply to that signal's exception,
the process can be left stopped after the detach. The inferior runs at
background priority to make that more likely, but the race is still caught
only probabilistically: a regression fails some runs, not every run.
"""

import time

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

# How long the heartbeat file may take to grow after a detach. The inferior
# runs at background priority and can wait a long time for a CPU on a busy
# machine. A process left stopped never grows the file.
HEARTBEAT_DEADLINE_S = 60


class DetachPendingSigstopTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def launch_inferior(self):
        """Build and launch the inferior; return its pid and heartbeat path."""
        self.build()

        # The inferior appends a line to this file every 100 ms. The file
        # appearing tells us it is ready to be attached to, and the file
        # growing after a detach tells us it is running again.
        heartbeat_path = lldbutil.append_to_process_working_directory(
            self, "heartbeat_%d" % (int(time.time()))
        )
        self.addTearDownHook(
            lambda: self.run_platform_command("rm %s" % (heartbeat_path))
        )

        popen = self.spawnSubprocess(self.getBuildArtifact("a.out"), [heartbeat_path])
        lldbutil.wait_for_file_on_target(self, heartbeat_path)
        return popen.pid, heartbeat_path

    def heartbeat_size(self, heartbeat_path):
        return len(lldbutil.read_file_on_target(self, heartbeat_path))

    def assert_heartbeat_grows_after_detach(self, heartbeat_path):
        """Check that the inferior is running: the file grows within
        HEARTBEAT_DEADLINE_S.

        The baseline is taken only now, after "process detach" has returned.
        Growth before that point, including during the brief resume debugserver
        uses to deliver its SIGSTOP, never counts.
        """
        baseline = self.heartbeat_size(heartbeat_path)
        deadline = time.monotonic() + HEARTBEAT_DEADLINE_S
        while time.monotonic() < deadline:
            if self.heartbeat_size(heartbeat_path) > baseline:
                return
            time.sleep(0.1)
        self.fail(
            "heartbeat file did not grow past %d bytes within %d s of detach"
            % (baseline, HEARTBEAT_DEADLINE_S)
        )

    @requireDarwin
    @skipIfOutOfTreeDebugserver
    @skipIfAsan
    def test_detach_after_expression(self):
        """Attach, run an expression and detach, ten times over."""
        pid, heartbeat_path = self.launch_inferior()

        # expect_expr's defaults, plus a timeout as long as the heartbeat
        # deadline: the inferior runs at background priority, so on a busy
        # machine the call can take far longer than the default 500 ms.
        options = lldb.SBExpressionOptions()
        options.SetAutoApplyFixIts(False)
        options.SetIgnoreBreakpoints(True)
        options.SetTimeoutInMicroSeconds(HEARTBEAT_DEADLINE_S * 1000000)

        for cycle in range(10):
            self.runCmd("process attach -p %d" % pid)
            # marker_function() counts its calls, so the value proves the
            # expression ran in the inferior on this cycle.
            self.expect_expr(
                "(int)marker_function()",
                result_type="int",
                result_value=str(cycle + 1),
                options=options,
            )
            self.runCmd("process detach")
            self.assert_heartbeat_grows_after_detach(heartbeat_path)

    @requireDarwin
    @skipIfOutOfTreeDebugserver
    @skipIfAsan
    def test_detach_after_attach(self):
        """Attach and detach, three times over."""
        pid, heartbeat_path = self.launch_inferior()

        for _ in range(3):
            self.runCmd("process attach -p %d" % pid)
            self.runCmd("process detach")
            self.assert_heartbeat_grows_after_detach(heartbeat_path)
