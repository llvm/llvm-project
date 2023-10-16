"""
Test lldb-dap RestartRequest.
"""

import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase


class TestDAP_restart_runInTerminal(lldbdap_testcase.DAPTestCaseBase):
    def isTestSupported(self):
        try:
            # We skip this test for debug builds because it takes too long
            # parsing lldb's own debug info. Release builds are fine.
            # Checking the size of the lldb-dap binary seems to be a decent
            # proxy for a quick detection. It should be far less than 1 MB in
            # Release builds.
            return os.path.getsize(os.environ["LLDBDAP_EXEC"]) < 1000000
        except:
            return False

    @skipIfWindows
    @skipIfRemote
    @skipIf(archs=["arm"])  # Always times out on buildbot
    def test_basic_functionality(self):
        """
        Test basic restarting functionality when the process is running in
        a terminal.
        """
        if not self.isTestSupported():
            return
        line_A = line_number("main.c", "// breakpoint A")
        line_B = line_number("main.c", "// breakpoint B")

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, runInTerminal=True)
        [bp_A, bp_B] = self.set_source_breakpoints("main.c", [line_A, line_B])

        # Verify we hit A, then B.
        self.dap.request_configurationDone()
        self.verify_breakpoint_hit([bp_A])
        self.dap.request_continue()
        self.verify_breakpoint_hit([bp_B])

        # Make sure i has been modified from its initial value of 0.
        self.assertEquals(
            int(self.dap.get_local_variable_value("i")),
            1234,
            "i != 1234 after hitting breakpoint B",
        )

        # Restart.
        self.dap.request_restart()

        # Finally, check we stop back at A and program state has been reset.
        self.verify_breakpoint_hit([bp_A])
        self.assertEquals(
            int(self.dap.get_local_variable_value("i")),
            0,
            "i != 0 after hitting breakpoint A on restart",
        )

    @skipIfWindows
    @skipIfRemote
    @skipIf(archs=["arm"])  # Always times out on buildbot
    def test_stopOnEntry(self):
        """
        Check that stopOnEntry works correctly when using runInTerminal.
        """
        if not self.isTestSupported():
            return
        line_A = line_number("main.c", "// breakpoint A")
        line_B = line_number("main.c", "// breakpoint B")

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, runInTerminal=True, stopOnEntry=True)
        [bp_main] = self.set_function_breakpoints(["main"])
        self.dap.request_configurationDone()

        # When using stopOnEntry, configurationDone doesn't result in a running
        # process, we should immediately get a stopped event instead.
        stopped_events = self.dap.wait_for_stopped()
        # We should be stopped at the entry point.
        for stopped_event in stopped_events:
            if "body" in stopped_event:
                body = stopped_event["body"]
                if "reason" in body:
                    reason = body["reason"]
                    self.assertNotEqual(
                        reason, "breakpoint", "verify stop isn't a breakpoint"
                    )

        # Then, if we continue, we should hit the breakpoint at main.
        self.dap.request_continue()
        self.verify_breakpoint_hit([bp_main])

        # Restart and check that we still get a stopped event before reaching
        # main.
        self.dap.request_restart()
        stopped_events = self.dap.wait_for_stopped()
        for stopped_event in stopped_events:
            if "body" in stopped_event:
                body = stopped_event["body"]
                if "reason" in body:
                    reason = body["reason"]
                    self.assertNotEqual(
                        reason,
                        "breakpoint",
                        'verify stop after restart isn\'t "main" breakpoint',
                    )
