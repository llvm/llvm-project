"""
Test lldb-dap RestartRequest.
"""

from typing import Dict, Any, List

import lldbdap_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number


@skipIfBuildType(["debug"])
class TestDAP_restart_console(lldbdap_testcase.DAPTestCaseBase):
    def verify_stopped_on_entry(self, stopped_events: List[Dict[str, Any]]):
        seen_stopped_event = 0
        for stopped_event in stopped_events:
            body = stopped_event.get("body")
            if body is None:
                continue

            reason = body.get("reason")
            if reason is None:
                continue

            self.assertNotEqual(
                reason,
                "breakpoint",
                'verify stop after restart isn\'t "main" breakpoint',
            )
            if reason == "entry":
                seen_stopped_event += 1

        self.assertEqual(seen_stopped_event, 1, "expect only one stopped entry event.")

    @skipIfAsan
    @skipIfWindows
    @skipIf(oslist=["linux"], archs=["arm$"])  # Always times out on buildbot
    def test_basic_functionality(self):
        """
        Test basic restarting functionality when the process is running in
        a terminal.
        """
        line_A = line_number("main.c", "// breakpoint A")
        line_B = line_number("main.c", "// breakpoint B")

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, console="integratedTerminal")
        [bp_A, bp_B] = self.set_source_breakpoints("main.c", [line_A, line_B])

        # Verify we hit A, then B.
        self.dap_server.request_configurationDone()
        self.verify_breakpoint_hit([bp_A])
        self.dap_server.request_continue()
        self.verify_breakpoint_hit([bp_B])

        # Make sure i has been modified from its initial value of 0.
        self.assertEqual(
            int(self.dap_server.get_local_variable_value("i")),
            1234,
            "i != 1234 after hitting breakpoint B",
        )

        # Restart.
        self.dap_server.request_restart()

        # Finally, check we stop back at A and program state has been reset.
        self.verify_breakpoint_hit([bp_A])
        self.assertEqual(
            int(self.dap_server.get_local_variable_value("i")),
            0,
            "i != 0 after hitting breakpoint A on restart",
        )

        # Check breakpoint B
        self.dap_server.request_continue()
        self.verify_breakpoint_hit([bp_B])
        self.assertEqual(
            int(self.dap_server.get_local_variable_value("i")),
            1234,
            "i != 1234 after hitting breakpoint B",
        )
        self.continue_to_exit()

    @skipIfAsan
    @skipIfWindows
    @skipIf(oslist=["linux"], archs=["arm$"])  # Always times out on buildbot
    def test_stopOnEntry(self):
        """
        Check that stopOnEntry works correctly when using console.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, console="integratedTerminal", stopOnEntry=True)
        [bp_main] = self.set_function_breakpoints(["main"])

        self.dap_server.request_continue()  # sends configuration done
        stopped_events = self.dap_server.wait_for_stopped()
        # We should be stopped at the entry point.
        self.assertGreaterEqual(len(stopped_events), 0, "expect stopped events")
        self.verify_stopped_on_entry(stopped_events)

        # Then, if we continue, we should hit the breakpoint at main.
        self.dap_server.request_continue()
        self.verify_breakpoint_hit([bp_main])

        # Restart and check that we still get a stopped event before reaching
        # main.
        self.dap_server.request_restart()
        stopped_events = self.dap_server.wait_for_stopped()
        self.verify_stopped_on_entry(stopped_events)

        # continue to main
        self.dap_server.request_continue()
        self.verify_breakpoint_hit([bp_main])

        self.continue_to_exit()
