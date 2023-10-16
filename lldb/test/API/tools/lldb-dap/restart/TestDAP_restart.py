"""
Test lldb-dap RestartRequest.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase


class TestDAP_restart(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    @skipIfRemote
    def test_basic_functionality(self):
        """
        Tests the basic restarting functionality: set two breakpoints in
        sequence, restart at the second, check that we hit the first one.
        """
        line_A = line_number("main.c", "// breakpoint A")
        line_B = line_number("main.c", "// breakpoint B")

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
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

        # Restart then check we stop back at A and program state has been reset.
        self.dap.request_restart()
        self.verify_breakpoint_hit([bp_A])
        self.assertEquals(
            int(self.dap.get_local_variable_value("i")),
            0,
            "i != 0 after hitting breakpoint A on restart",
        )

    @skipIfWindows
    @skipIfRemote
    def test_stopOnEntry(self):
        """
        Check that the stopOnEntry setting is still honored after a restart.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        [bp_main] = self.set_function_breakpoints(["main"])
        self.dap.request_configurationDone()

        # Once the "configuration done" event is sent, we should get a stopped
        # event immediately because of stopOnEntry.
        stopped_events = self.dap.wait_for_stopped()
        for stopped_event in stopped_events:
            if "body" in stopped_event:
                body = stopped_event["body"]
                if "reason" in body:
                    reason = body["reason"]
                    self.assertNotEqual(
                        reason, "breakpoint", 'verify stop isn\'t "main" breakpoint'
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

    @skipIfWindows
    @skipIfRemote
    def test_arguments(self):
        """
        Tests that lldb-dap will use updated launch arguments included
        with a restart request.
        """
        line_A = line_number("main.c", "// breakpoint A")

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        [bp_A] = self.set_source_breakpoints("main.c", [line_A])

        # Verify we hit A, then B.
        self.dap.request_configurationDone()
        self.verify_breakpoint_hit([bp_A])

        # We don't set any arguments in the initial launch request, so argc
        # should be 1.
        self.assertEquals(
            int(self.dap.get_local_variable_value("argc")),
            1,
            "argc != 1 before restart",
        )

        # Restart with some extra 'args' and check that the new argc reflects
        # the updated launch config.
        self.dap.request_restart(
            restartArguments={
                "arguments": {
                    "program": program,
                    "args": ["a", "b", "c", "d"],
                }
            }
        )
        self.verify_breakpoint_hit([bp_A])
        self.assertEquals(
            int(self.dap.get_local_variable_value("argc")),
            5,
            "argc != 5 after restart",
        )
