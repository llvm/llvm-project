"""
Test lldb-dap setBreakpoints request
"""

import dap_server
import lldbdap_testcase
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


def get_subprocess(root_process, process_name):
    queue = [root_process]
    while queue:
        process = queue.pop()
        if process.name() == process_name:
            return process
        queue.extend(process.children())

    self.assertTrue(False, "No subprocess with name %s found" % process_name)

class TestDAP_console(lldbdap_testcase.DAPTestCaseBase):
    def check_lldb_command(
        self, lldb_command, contains_string, assert_msg, command_escape_prefix="`"
    ):
        response = self.dap_server.request_evaluate(
            f"{command_escape_prefix}{lldb_command}", context="repl"
        )
        output = response["body"]["result"]
        self.assertIn(
            contains_string,
            output,
            (
                """Verify %s by checking the command output:\n"""
                """'''\n%s'''\nfor the string: "%s" """
                % (assert_msg, output, contains_string)
            ),
        )

    def test_scopes_variables_setVariable_evaluate(self):
        """
        Tests that the "scopes" request causes the currently selected
        thread and frame to be updated. There are no DAP packets that tell
        lldb-dap which thread and frame are selected other than the
        "scopes" request. lldb-dap will now select the thread and frame
        for the latest "scopes" request that it receives.

        The LLDB command interpreter needs to have the right thread and
        frame selected so that commands executed in the debug console act
        on the right scope. This applies both to the expressions that are
        evaluated and the lldb commands that start with the backtick
        character.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        lines = [breakpoint1_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)
        # Cause a "scopes" to be sent for frame zero which should update the
        # selected thread and frame to frame 0.
        self.dap_server.get_local_variables(frameIndex=0)
        # Verify frame #0 is selected in the command interpreter by running
        # the "frame select" command with no frame index which will print the
        # currently selected frame.
        self.check_lldb_command("frame select", "frame #0", "frame 0 is selected")

        # Cause a "scopes" to be sent for frame one which should update the
        # selected thread and frame to frame 1.
        self.dap_server.get_local_variables(frameIndex=1)
        # Verify frame #1 is selected in the command interpreter by running
        # the "frame select" command with no frame index which will print the
        # currently selected frame.

        self.check_lldb_command("frame select", "frame #1", "frame 1 is selected")

    def test_custom_escape_prefix(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, commandEscapePrefix="::")
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        breakpoint_ids = self.set_source_breakpoints(source, [breakpoint1_line])
        self.continue_to_breakpoints(breakpoint_ids)

        self.check_lldb_command(
            "help",
            "For more information on any command",
            "Help can be invoked",
            command_escape_prefix="::",
        )

    def test_empty_escape_prefix(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, commandEscapePrefix="")
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        breakpoint_ids = self.set_source_breakpoints(source, [breakpoint1_line])
        self.continue_to_breakpoints(breakpoint_ids)

        self.check_lldb_command(
            "help",
            "For more information on any command",
            "Help can be invoked",
            command_escape_prefix="",
        )

    @skipIfWindows
    def test_exit_status_message_sigterm(self):
        source = "main.cpp"
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, commandEscapePrefix="")
        breakpoint1_line = line_number(source, "// breakpoint 1")
        breakpoint_ids = self.set_source_breakpoints(source, [breakpoint1_line])
        self.continue_to_breakpoints(breakpoint_ids)

        # Kill lldb-server process.
        process_name = (
            "debugserver" if platform.system() in ["Darwin"] else "lldb-server"
        )

        try:
            import psutil
        except ImportError:
            print(
                "psutil not installed, please install using 'pip install psutil'. "
                "Skipping test_exit_status_message_sigterm test.",
                file=sys.stderr,
            )
            return
        process = get_subprocess(psutil.Process(os.getpid()), process_name)
        process.terminate()
        process.wait()

        # Get the console output
        console_output = self.collect_console(
            timeout_secs=10.0, pattern="exited with status"
        )

        # Verify the exit status message is printed.
        self.assertIn(
            "exited with status = -1 (0xffffffff) debugserver died with signal SIGTERM",
            console_output,
            "Exit status does not contain message 'exited with status'",
        )

    def test_exit_status_message_ok(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, commandEscapePrefix="")
        self.continue_to_exit()

        # Get the console output
        console_output = self.collect_console(
            timeout_secs=10.0, pattern="exited with status"
        )

        # Verify the exit status message is printed.
        self.assertIn(
            "exited with status = 0 (0x00000000)",
            console_output,
            "Exit status does not contain message 'exited with status'",
        )
