"""
Test lldb-dap RestartRequest.
"""

from typing import Dict, Any, List

import lldbdap_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number


@skipIfBuildType(["debug"])
class TestDAP_restart_console(lldbdap_testcase.DAPTestCaseBase):
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

        self.dap_server.request_configurationDone()
        self.verify_stop_on_entry()

        # Then, if we continue, we should hit the breakpoint at main.
        self.dap_server.request_continue()
        self.verify_breakpoint_hit([bp_main])

        # Restart and check that we still get a stopped event before reaching
        # main.
        self.dap_server.request_restart()
        self.verify_stop_on_entry()

        # continue to main
        self.dap_server.request_continue()
        self.verify_breakpoint_hit([bp_main])

        self.continue_to_exit()
