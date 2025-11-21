"""
Test lldb-dap RestartRequest.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase


class TestDAP_restart(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
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
        self.continue_to_breakpoints([bp_A])
        self.continue_to_breakpoints([bp_B])

        # Make sure i has been modified from its initial value of 0.
        self.assertEqual(
            int(self.dap_server.get_local_variable_value("i")),
            1234,
            "i != 1234 after hitting breakpoint B",
        )

        # Restart then check we stop back at A and program state has been reset.
        resp = self.dap_server.request_restart()
        self.assertTrue(resp["success"])
        self.verify_breakpoint_hit([bp_A])
        self.assertEqual(
            int(self.dap_server.get_local_variable_value("i")),
            0,
            "i != 0 after hitting breakpoint A on restart",
        )

    @skipIfWindows
    def test_stopOnEntry(self):
        """
        Check that the stopOnEntry setting is still honored after a restart.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        [bp_main] = self.set_function_breakpoints(["main"])

        self.continue_to_next_stop()
        self.verify_stop_on_entry()

        # Then, if we continue, we should hit the breakpoint at main.
        self.continue_to_breakpoints([bp_main])

        # Restart and check that we still get a stopped event before reaching
        # main.
        resp = self.dap_server.request_restart()
        self.assertTrue(resp["success"])
        self.verify_stop_on_entry()

    @skipIfWindows
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
        self.continue_to_breakpoints([bp_A])

        # We don't set any arguments in the initial launch request, so argc
        # should be 1.
        self.assertEqual(
            int(self.dap_server.get_local_variable_value("argc")),
            1,
            "argc != 1 before restart",
        )

        # Restart with some extra 'args' and check that the new argc reflects
        # the updated launch config.
        resp = self.dap_server.request_restart(
            restartArguments={
                "arguments": {
                    "program": program,
                    "args": ["a", "b", "c", "d"],
                }
            }
        )
        self.assertTrue(resp["success"])
        self.verify_breakpoint_hit([bp_A])
        self.assertEqual(
            int(self.dap_server.get_local_variable_value("argc")),
            5,
            "argc != 5 after restart",
        )
