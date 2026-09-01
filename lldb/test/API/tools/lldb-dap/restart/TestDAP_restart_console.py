"""
Test lldb-dap RestartRequest.
"""

from lldbsuite.test.decorators import (
    skipIf,
    skipIfAsan,
    skipIfBuildType,
    skipIfWasm,
    skipIfWindows,
)
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import Console, LaunchArgs


@skipIfBuildType(["debug"])
@skipIfWasm  # runInTerminal has the client run the program, and a Wasm module is not executable
class TestDAP_restart_console(DAPTestCaseBase):
    @skipIfAsan
    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/200840
    @skipIf(oslist=["linux"], archs=["arm$"])  # Always times out on buildbot
    def test_basic_functionality(self):
        """
        Test basic restarting functionality when the process is running in
        a terminal.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        launch_args = LaunchArgs(program, console=Console.INTEGRATED_TERMINAL)
        with session.configure(launch_args) as ctx:
            line_A = line_number("main.c", "// breakpoint A")
            line_B = line_number("main.c", "// breakpoint B")

            [bp_A, bp_B] = session.resolve_source_breakpoints(
                "main.c", [line_A, line_B]
            )

        # Verify we hit A, then B.
        stop_A = session.verify_stopped_on_breakpoint(bp_A, after=ctx.process_event)
        session.do_continue()
        stop_B = session.verify_stopped_on_breakpoint(bp_B, after=stop_A)

        # Make sure i has been modified from its initial value of 0.
        top_frame = session.top_frame_from(stop_B)
        self.assertEqual(
            top_frame.locals["i"].value_as_int,
            1234,
            "i != 1234 after hitting breakpoint B",
        )

        last_event = session.last_event()
        # Restart.
        session.restart()

        # Finally, check we stop back at A and program state has been reset.
        stop_A = session.verify_stopped_on_breakpoint(bp_A, after=last_event)
        top_frame = session.top_frame_from(stop_A)
        i_val = top_frame.locals["i"].value_as_int
        self.assertEqual(i_val, 0, "i != 0 after hitting breakpoint A on restart")

        # Check breakpoint B.
        session.do_continue()
        stop_B = session.verify_stopped_on_breakpoint(bp_B, after=stop_A)
        top_frame = session.top_frame_from(stop_B)
        self.assertEqual(
            top_frame.locals["i"].value_as_int,
            1234,
            "i != 1234 after hitting breakpoint B",
        )
        session.continue_to_exit()

    @skipIfAsan
    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/200840
    @skipIf(oslist=["linux"], archs=["arm$"])  # Always times out on buildbot
    def test_stopOnEntry(self):
        """
        Check that stopOnEntry works correctly when using console.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        launch_args = LaunchArgs(
            program, console=Console.INTEGRATED_TERMINAL, stopOnEntry=True
        )
        with session.configure(launch_args) as ctx:
            [bp_main] = session.resolve_function_breakpoints(["main"])
        session.verify_stopped_on_entry(after=ctx.process_event)

        # Then, if we continue, we should hit the breakpoint at main.
        stop_event = session.continue_to_breakpoint(bp_main)

        # Restart and check that we still get a stopped event before reaching
        # main.
        session.restart()
        session.verify_stopped_on_entry(after=stop_event)

        # continue to main
        session.continue_to_breakpoint(bp_main)
        session.continue_to_exit()
