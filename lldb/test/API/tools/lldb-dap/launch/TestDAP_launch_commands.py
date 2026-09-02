"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIf
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_commands(DAPTestCaseBase):
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
        session = self.build_and_create_session()

        launch_args = LaunchArgs(
            program,
            initCommands=initCommands,
            preRunCommands=preRunCommands,
            postRunCommands=postRunCommands,
            stopCommands=stopCommands,
            exitCommands=exitCommands,
            terminateCommands=terminateCommands,
        )
        with session.configure(launch_args) as ctx:
            # Get output from the console. This should contain the
            # "initCommands", "preRunCommands", and "postRunCommands".
            coutput = session.collect_console(
                after=ctx.init_response, until=postRunCommands[-1]
            )
            output = coutput.seen_texts
            session.verify_commands("initCommands", output, initCommands)
            session.verify_commands("preRunCommands", output, preRunCommands)
            session.verify_commands("postRunCommands", output, postRunCommands)

            source = "main.c"
            first_line = line_number(source, "// breakpoint 1")
            second_line = line_number(source, "// breakpoint 2")
            lines = [first_line, second_line]

            # Set 2 breakpoints so we can verify that "stopCommands" get run as the
            # breakpoints get hit.
            [first_bp, second_bp] = session.resolve_source_breakpoints(source, lines)

        launch_response = ctx.launch_or_attach_response

        # Continue after launch and hit the first breakpoint.
        # Get output from the console. This should contain the
        # "stopCommands" that were run after the first breakpoint was hit.
        session.verify_stopped_on_breakpoint(first_bp, after=ctx.process_event)
        coutput = session.collect_console(after=launch_response, until=stopCommands[-1])
        output = coutput.seen_texts
        session.verify_commands("stopCommands", output, stopCommands)

        # Continue again and hit the second breakpoint.
        # Get output from the console. This should contain the
        # "stopCommands" that were run after the second breakpoint was hit.
        session.continue_to_breakpoint(second_bp)
        coutput = session.collect_console(after=coutput.event, until=stopCommands[-1])
        output = coutput.seen_texts
        session.verify_commands("stopCommands", output, stopCommands)

        # Continue until the program exits.
        # Get output from the console. This should contain the
        # "exitCommands" run on process exit and the "terminateCommands"
        # run when the debugging session ends.
        session.continue_to_exit()
        coutput = session.collect_console(
            after=coutput.event, until=terminateCommands[0]
        )
        output = coutput.seen_texts
        session.verify_commands("exitCommands", output, exitCommands)
        session.verify_commands("terminateCommands", output, terminateCommands)
