"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIf, skipIfNetBSD
from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_terminate_commands(DAPTestCaseBase):
    """
    Tests that the "terminateCommands", that can be passed during launch,
    are run when the debugger is disconnected.
    """

    @skipIfNetBSD  # Hangs on NetBSD as well
    @skipIf(archs=["arm$", "aarch64"], oslist=["linux"])
    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session(disconnect_automatically=False)

        terminate_commands = ["history"]
        process_event = session.launch(
            LaunchArgs(
                program=program,
                stopOnEntry=True,
                terminateCommands=terminate_commands,
            )
        )
        stop_event = session.verify_stopped_on_entry(after=process_event)
        # Once it's disconnected the console should contain the "terminateCommands".
        session.disconnect(terminateDebuggee=True)
        output = session.collect_console(after=stop_event, until=terminate_commands[0])
        session.verify_commands(
            "terminateCommands", output.seen_texts, terminate_commands
        )
