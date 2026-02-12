"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIf
from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase


class TestDAP_launch_commands(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the "initCommands", "preRunCommands", "stopCommands",
    "terminateCommands" and "exitCommands" that can be passed during
    launch.

    "initCommands" are a list of LLDB commands that get executed
    before the target is created.
    "preRunCommands" are a list of LLDB commands that get executed
    after the target has been created and before the launch.
    "stopCommands" are a list of LLDB commands that get executed each
    time the program stops.
    "exitCommands" are a list of LLDB commands that get executed when
    the process exits
    "terminateCommands" are a list of LLDB commands that get executed when
    the debugger session terminates.
    """

    @skipIf(archs=["arm$", "aarch64"], bugnumber=6933)
    def test(self):
        program = self.getBuildArtifact("a.out")
        initCommands = ["target list", "platform list"]
        preRunCommands = ["image list a.out", "image dump sections a.out"]
        postRunCommands = ["help trace", "help process trace"]
        stopCommands = ["frame variable", "bt"]
        exitCommands = ["expr 2+3", "expr 3+4"]
        terminateCommands = ["expr 4+2"]
        self.build_and_launch(
            program,
            initCommands=initCommands,
            preRunCommands=preRunCommands,
            postRunCommands=postRunCommands,
            stopCommands=stopCommands,
            exitCommands=exitCommands,
            terminateCommands=terminateCommands,
        )
        self.dap_server.wait_for_initialized()

        # Get output from the console. This should contain both the
        # "initCommands" and the "preRunCommands".
        output = self.collect_console(pattern=postRunCommands[-1])
        # Verify all "initCommands" were found in console output
        self.verify_commands("initCommands", output, initCommands)
        # Verify all "preRunCommands" were found in console output
        self.verify_commands("preRunCommands", output, preRunCommands)
        # Verify all "postRunCommands" were found in console output
        self.verify_commands("postRunCommands", output, postRunCommands)

        source = "main.c"
        first_line = line_number(source, "// breakpoint 1")
        second_line = line_number(source, "// breakpoint 2")
        lines = [first_line, second_line]

        # Set 2 breakpoints so we can verify that "stopCommands" get run as the
        # breakpoints get hit
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )

        # Continue after launch and hit the first breakpoint.
        # Get output from the console. This should contain both the
        # "stopCommands" that were run after the first breakpoint was hit
        self.continue_to_breakpoints(breakpoint_ids)
        output = self.collect_console(pattern=stopCommands[-1])
        self.verify_commands("stopCommands", output, stopCommands)

        # Continue again and hit the second breakpoint.
        # Get output from the console. This should contain both the
        # "stopCommands" that were run after the second breakpoint was hit
        self.continue_to_breakpoints(breakpoint_ids)
        output = self.collect_console(pattern=stopCommands[-1])
        self.verify_commands("stopCommands", output, stopCommands)

        # Continue until the program exits
        self.continue_to_exit()
        # Get output from the console. This should contain both the
        # "exitCommands" that were run after the second breakpoint was hit
        # and the "terminateCommands" due to the debugging session ending
        output = self.collect_console(pattern=terminateCommands[0])
        self.verify_commands("exitCommands", output, exitCommands)
        self.verify_commands("terminateCommands", output, terminateCommands)
