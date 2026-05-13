"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIf
from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase


class TestDAP_launch_extra_launch_commands(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the "launchCommands" with extra launching settings
    """

    # Flakey on 32-bit Arm Linux.
    @skipIf(oslist=["linux"], archs=["arm$"])
    def test(self):
        self.build_and_create_debug_adapter()
        program = self.getBuildArtifact("a.out")

        source = "main.c"
        first_line = line_number(source, "// breakpoint 1")
        second_line = line_number(source, "// breakpoint 2")
        # Set target binary and 2 breakpoints
        # then we can verify the "launchCommands" get run
        # also we can verify that "stopCommands" get run as the
        # breakpoints get hit
        launchCommands = [
            'target create "%s"' % (program),
            "process launch --stop-at-entry",
        ]
        initCommands = ["target list", "platform list"]
        preRunCommands = ["image list a.out", "image dump sections a.out"]
        postRunCommands = ['script print("hello world")']
        stopCommands = ["frame variable", "bt"]
        exitCommands = ["expr 2+3", "expr 3+4"]
        self.launch(
            program,
            initCommands=initCommands,
            preRunCommands=preRunCommands,
            postRunCommands=postRunCommands,
            stopCommands=stopCommands,
            exitCommands=exitCommands,
            launchCommands=launchCommands,
        )
        self.set_source_breakpoints("main.c", [first_line, second_line])

        # Get output from the console. This should contain both the
        # "initCommands" and the "preRunCommands".
        output = self.get_console()
        # Verify all "initCommands" were found in console output
        self.verify_commands("initCommands", output, initCommands)
        # Verify all "preRunCommands" were found in console output
        self.verify_commands("preRunCommands", output, preRunCommands)

        # Verify all "launchCommands" were found in console output
        # After execution, program should launch
        self.verify_commands("launchCommands", output, launchCommands)
        self.verify_commands("postRunCommands", output, postRunCommands)
        # Verify the "stopCommands" here
        self.continue_to_next_stop()
        output = self.get_console()
        self.verify_commands("stopCommands", output, stopCommands)

        # Continue and hit the second breakpoint.
        # Get output from the console. This should contain both the
        # "stopCommands" that were run after the first breakpoint was hit
        self.continue_to_next_stop()
        output = self.get_console()
        self.verify_commands("stopCommands", output, stopCommands)

        # Continue until the program exits
        self.continue_to_exit()
        # Get output from the console. This should contain both the
        # "exitCommands" that were run after the second breakpoint was hit
        output = self.get_console()
        self.verify_commands("exitCommands", output, exitCommands)
