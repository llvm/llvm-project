"""
Test lldb-vscode setBreakpoints request
"""


import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import os


class TestVSCode_breakpointEvents(lldbvscode_testcase.VSCodeTestCaseBase):
    @skipIfWindows
    @skipUnlessDarwin
    @expectedFailureAll(macos_version=[">=", "13.0"])
    def test_breakpoint_events(self):
        """
        This test sets a breakpoint in a shared library and runs and stops
        at the entry point of a program. When we stop at the entry point,
        the shared library won't be loaded yet. At this point the
        breakpoint should set itself, but not be verified because no
        locations are resolved. We will then continue and expect to get a
        breakpoint event that informs us that the breakpoint in the shared
        library is "changed" and the correct line number should be
        supplied. We also set a breakpoint using a LLDB command using the
        "preRunCommands" when launching our program. Any breakpoints set via
        the command interpreter should not be have breakpoint events sent
        back to VS Code as the UI isn't able to add new breakpoints to
        their UI. Code has been added that tags breakpoints set from VS Code
        DAP packets so we know the IDE knows about them. If VS Code is ever
        able to register breakpoints that aren't initially set in the GUI,
        then we will need to revise this.
        """
        main_source_basename = "main.cpp"
        main_source_path = os.path.join(os.getcwd(), main_source_basename)
        foo_source_basename = "foo.cpp"
        foo_source_path = os.path.join(os.getcwd(), foo_source_basename)
        main_bp_line = line_number("main.cpp", "main breakpoint 1")
        foo_bp1_line = line_number("foo.cpp", "foo breakpoint 1")
        foo_bp2_line = line_number("foo.cpp", "foo breakpoint 2")

        # Visual Studio Code Debug Adaptors have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")

        # Set a breakpoint after creating the target by running a command line
        # command. It will eventually resolve and cause a breakpoint changed
        # event to be sent to lldb-vscode. We want to make sure we don't send a
        # breakpoint any breakpoints that were set from the command line.
        # Breakpoints that are set via the VS code DAP packets will be
        # registered and marked with a special keyword to ensure we deliver
        # breakpoint events for these breakpoints but not for ones that are not
        # set via the command interpreter.
        bp_command = "breakpoint set --file foo.cpp --line %u" % (foo_bp2_line)
        self.build_and_launch(program, stopOnEntry=True, preRunCommands=[bp_command])
        main_bp_id = 0
        foo_bp_id = 0
        # Set breakpoints and verify that they got set correctly
        vscode_breakpoint_ids = []
        response = self.vscode.request_setBreakpoints(main_source_path, [main_bp_line])
        if response:
            breakpoints = response["body"]["breakpoints"]
            for breakpoint in breakpoints:
                main_bp_id = breakpoint["id"]
                vscode_breakpoint_ids.append("%i" % (main_bp_id))
                # line = breakpoint['line']
                self.assertTrue(
                    breakpoint["verified"], "expect main breakpoint to be verified"
                )

        response = self.vscode.request_setBreakpoints(foo_source_path, [foo_bp1_line])
        if response:
            breakpoints = response["body"]["breakpoints"]
            for breakpoint in breakpoints:
                foo_bp_id = breakpoint["id"]
                vscode_breakpoint_ids.append("%i" % (foo_bp_id))
                self.assertFalse(
                    breakpoint["verified"], "expect foo breakpoint to not be verified"
                )

        # Get the stop at the entry point
        self.continue_to_next_stop()

        # We are now stopped at the entry point to the program. Shared
        # libraries are not loaded yet (at least on macOS they aren't) and any
        # breakpoints set in foo.cpp should not be resolved.
        self.assertEqual(
            len(self.vscode.breakpoint_events),
            0,
            "no breakpoint events when stopped at entry point",
        )

        # Continue to the breakpoint
        self.continue_to_breakpoints(vscode_breakpoint_ids)

        # Make sure we only get an event for the breakpoint we set via a call
        # to self.vscode.request_setBreakpoints(...), not the breakpoint
        # we set with with a LLDB command in preRunCommands.
        self.assertEqual(
            len(self.vscode.breakpoint_events), 1, "make sure we got a breakpoint event"
        )
        event = self.vscode.breakpoint_events[0]
        # Verify the details of the breakpoint changed notification.
        body = event["body"]
        self.assertEqual(
            body["reason"], "changed", "breakpoint event is says breakpoint is changed"
        )
        breakpoint = body["breakpoint"]
        self.assertTrue(
            breakpoint["verified"], "breakpoint event is says it is verified"
        )
        self.assertEqual(
            breakpoint["id"],
            foo_bp_id,
            "breakpoint event is for breakpoint %i" % (foo_bp_id),
        )
        self.assertTrue(
            "line" in breakpoint and breakpoint["line"] > 0,
            "breakpoint event is has a line number",
        )
        self.assertNotIn(
            "source", breakpoint, "breakpoint event should not return a source object"
        )
