"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIf
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_extra_launch_commands(DAPTestCaseBase):
    """
    Tests the "launchCommands" with extra launching settings
    """

    # Flakey on 32-bit Arm Linux.
    @skipIf(oslist=["linux"], archs=["arm$"])
    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.c")
        first_line = line_number(source, "// breakpoint 1")
        second_line = line_number(source, "// breakpoint 2")

        # Set target binary and 2 breakpoints, so we can verify the
        # "launchCommands" get run, and verify "stopCommands" get run
        # as the breakpoints get hit.
        launchCommands = [
            f'target create "{program}"',
            "process launch --stop-at-entry",
        ]
        initCommands = ["target list", "platform list"]
        preRunCommands = ["image list a.out", "image dump sections a.out"]
        postRunCommands = ['script print("hello world")']
        stopCommands = ["frame variable", "bt"]
        exitCommands = ["expr 2+3", "expr 3+4"]

        with session.configure(
            LaunchArgs(
                program=program,
                launchCommands=launchCommands,
                initCommands=initCommands,
                preRunCommands=preRunCommands,
                postRunCommands=postRunCommands,
                stopCommands=stopCommands,
                exitCommands=exitCommands,
            )
        ) as ctx:
            session.resolve_source_breakpoints(source, [first_line, second_line])
        process_event = ctx.process_event
        # The launchCommands stop the process at entry, but lldb-dap auto-continues
        # after configurationDone (since stopOnEntry isn't set), so the first
        # observable stop is hitting the first breakpoint, not entry.
        first_stop = session.verify_stopped_on_breakpoint(after=process_event)

        # Get output from the console. This should contain the
        # "initCommands", "preRunCommands", "launchCommands", and "postRunCommands".
        output = session.get_console()
        session.verify_commands("initCommands", output, initCommands)
        session.verify_commands("preRunCommands", output, preRunCommands)
        session.verify_commands("launchCommands", output, launchCommands)
        session.verify_commands("postRunCommands", output, postRunCommands)

        # Check that we got module events from target.
        session.wait_for_module_event(after=ctx.init_response)

        # Verify the "stopCommands" ran on the first breakpoint hit.
        # Wait until the last stopCommand's output arrives to avoid racing it.
        output = session.collect_console(after=first_stop, until=stopCommands[-1])
        session.verify_commands("stopCommands", output.seen_texts, stopCommands)

        # Continue and hit the second breakpoint, then verify "stopCommands"
        # ran again.
        session.continue_to_next_stop()
        output = session.collect_console(after=output.event, until=stopCommands[-1])
        session.verify_commands("stopCommands", output.seen_texts, stopCommands)

        # Continue until the program exits, then verify "exitCommands" ran.
        session.continue_to_exit()
        output = session.collect_console(after=output.event, until=exitCommands[-1])
        session.verify_commands("exitCommands", output.seen_texts, exitCommands)
