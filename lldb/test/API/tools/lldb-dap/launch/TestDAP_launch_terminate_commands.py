"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.tools.lldb_dap.types import ExitedEvent, LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_terminate_commands(DAPTestCaseBase):
    """
    Tests that the "exitCommands" and "terminateCommands", that can be passed
    during launch, are run when the debugger is disconnected.
    """

    @skipIfNetBSD  # Hangs on NetBSD as well
    @skipIf(archs=["arm$", "aarch64"], oslist=["linux"])
    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session(disconnect_automatically=False)

        exit_commands = ["version"]
        terminate_commands = ["history"]
        process_event = session.launch(
            LaunchArgs(
                program=program,
                stopOnEntry=True,
                exitCommands=exit_commands,
                terminateCommands=terminate_commands,
            )
        )
        stop_event = session.verify_stopped_on_entry(after=process_event)
        response = session.disconnect(terminateDebuggee=True)

        exited = session.wait_for_event(ExitedEvent, after=stop_event)
        terminated = session.wait_for_terminated_event(after=stop_event)
        self.assertLess(terminated.seq, response.seq)

        output = session.collect_console(after=stop_event, until=exited)
        session.verify_commands("exitCommands", output.seen_texts, exit_commands)
        output = session.collect_console(after=stop_event, until=terminated)
        session.verify_commands(
            "terminateCommands", output.seen_texts, terminate_commands
        )
