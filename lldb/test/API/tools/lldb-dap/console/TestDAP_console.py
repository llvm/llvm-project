"""
Test lldb-dap debug console output.
"""

import importlib.util
import os
import unittest

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.skip_reason import UnsupportedReason


requirePsutil = unittest.skipUnless(
    importlib.util.find_spec("psutil") is not None,
    UnsupportedReason(
        "psutil not installed, please install using 'pip install psutil'."
    ),
)


def get_subprocess(root_process, process_name: str):
    queue = [root_process]
    while queue:
        process = queue.pop()
        if process.name() == process_name:
            return process
        queue.extend(process.children())

    raise AssertionError(f"No subprocess with name {process_name!r} found")


class TestDAP_console(DAPTestCaseBase):
    def check_lldb_command(
        self,
        session: DAPTestSession,
        lldb_command: str,
        contains: str,
        escape_prefix: str = "`",
    ):
        resp_body = session.evaluate(f"{escape_prefix}{lldb_command}", context="repl")
        session.verify_evaluate(resp_body, matches=f".*{contains}.*")

    def do_test_with_escape_prefix(self, escape_prefix: str):
        """Launch with the given `commandEscapePrefix`, stop on the breakpoint,
        run `help` with the prefix, and exit."""
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")

        launch_args = LaunchArgs(program, commandEscapePrefix=escape_prefix)
        with session.configure(launch_args) as ctx:
            bp_ids = session.resolve_source_breakpoints(source, [breakpoint1_line])
        session.verify_stopped_on_breakpoint(bp_ids, after=ctx.process_event)

        self.check_lldb_command(
            session,
            "help",
            "For more information on any command",
            escape_prefix=escape_prefix,
        )
        session.continue_to_exit()

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
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        with session.configure(LaunchArgs(program)) as ctx:
            bp_ids = session.resolve_source_breakpoints(source, [breakpoint1_line])
        stop_event = session.verify_stopped_on_breakpoint(
            bp_ids, after=ctx.process_event
        )

        # Cause a "scopes" to be sent for frame zero which should update the
        # selected thread and frame to frame 0.
        thread_ctx = session.thread_context_from(stop_event)
        frame_ctxs = thread_ctx.frames()
        frame_ctxs[0].locals.variables()

        # Verify frame #0 is selected in the command interpreter by running
        # the "frame select" command with no frame index which will print the
        # currently selected frame.
        self.check_lldb_command(session, "frame select", "frame #0")

        # Cause a "scopes" to be sent for frame one which should update the
        # selected thread and frame to frame 1.
        frame_ctxs[1].locals.variables()

        # Verify frame #1 is selected in the command interpreter by running
        # the "frame select" command with no frame index which will print the
        # currently selected frame.
        self.check_lldb_command(session, "frame select", "frame #1")

        session.continue_to_exit()

    def test_custom_escape_prefix(self):
        self.do_test_with_escape_prefix("::")

    def test_empty_escape_prefix(self):
        self.do_test_with_escape_prefix("")

    @requireNotWindows("requires lldb-server")
    @requirePsutil
    @requireNotWasm("the test signals the debug server, which for Wasm is the runtime")
    def test_exit_status_message_sigterm(self):
        import psutil

        debug_server_path = self.get_debug_server_path()
        if debug_server_path is None:
            self.skipTest(f"{self.getPlatform()!r} does not have a debug server.")

        session = self.build_and_create_session()
        source = "main.cpp"
        program = self.getBuildArtifact("a.out")
        breakpoint1_line = line_number(source, "// breakpoint 1")
        with session.configure(LaunchArgs(program, commandEscapePrefix="")) as ctx:
            breakpoint_ids = session.resolve_source_breakpoints(
                source, [breakpoint1_line]
            )

        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_ids, after=ctx.process_event
        )

        # Kill lldb-server process.
        debug_server_name = debug_server_path.stem
        process = get_subprocess(psutil.Process(os.getpid()), debug_server_name)
        process.terminate()
        process.wait()

        # Get the console output
        captured = session.collect_console(after=stop_event, until="exited with status")

        # Verify the exit status message is printed.
        self.assertRegex(
            captured.seen_texts,
            ".*exited with status = -1 .* died with signal SIGTERM.*",
            "exit status does not contain message 'exited with status'",
        )

    def test_exit_status_message_ok(self):
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        process_event = session.launch(LaunchArgs(program, commandEscapePrefix=""))
        session.verify_process_exited()

        # Get the console output.
        captured = session.collect_console(
            after=process_event, until="exited with status"
        )

        # Verify the exit status message is printed.
        self.assertIn(
            "exited with status = 0 (0x00000000)",
            captured.seen_texts,
            "exit status does not contain message 'exited with status'",
        )

    def test_diagnostics(self):
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        process_event = session.launch(LaunchArgs(program, stopOnEntry=True))
        stop_event = session.verify_stopped_on_entry(after=process_event)

        core = self.getBuildArtifact("minidump.core")
        self.yaml2obj("minidump.yaml", core)
        session.evaluate(f"target create --core {core}", context="repl")

        captured = session.collect_important(after=stop_event, until="minidump file")

        self.assertIn(
            "warning: unable to retrieve process ID from minidump file",
            captured.seen_texts,
            "diagnostic found in important output",
        )
        session.continue_to_exit()
