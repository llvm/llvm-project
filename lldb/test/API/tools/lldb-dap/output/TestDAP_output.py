"""
Test lldb-dap output events
"""

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_output(DAPTestCaseBase):
    @skipIfWindows
    def test_output(self):
        """
        Test output handling for the running process.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session(disconnect_automatically=False)
        launch_args = LaunchArgs(
            program,
            exitCommands=[
                # Ensure that output produced by lldb itself is not consumed by the OutputRedirector.
                "?script print('out\\0\\0', end='\\r\\n', file=sys.stdout)",
                "?script print('err\\0\\0', end='\\r\\n', file=sys.stderr)",
            ],
        )
        source = "main.c"
        breakpoint_line = line_number(source, "// breakpoint 1")
        with session.configure(launch_args) as ctx:
            bp_ids = session.resolve_source_breakpoints(source, [breakpoint_line])

        process_event = ctx.process_event
        session.verify_stopped_on_breakpoint(bp_ids, after=process_event)

        # Ensure partial messages are still sent.
        partial_output = session.collect_stdout(after=process_event, until="abcdef")
        self.assertGreater(len(partial_output.seen_texts), 0, "expect program stdout")

        session.continue_to_exit()

        # Disconnecting from the server to ensure any pending IO is flushed.
        session.disconnect()

        stdout = session.get_stdout()
        self.assertTrue(stdout, "expect program stdout")
        self.assertIn(
            "abcdefghi\r\nhello world\r\nfinally\0\0",
            stdout,
            "full stdout not found in: " + repr(stdout),
        )
        console = session.get_console()
        self.assertTrue(console, "expect dap messages")
        self.assertIn(
            "out\0\0\r\nerr\0\0\r\n", console, "full console message not found"
        )
