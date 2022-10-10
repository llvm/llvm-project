"""
Test lldb-vscode setBreakpoints request
"""

import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase


class TestVSCode_console(lldbvscode_testcase.VSCodeTestCaseBase):

    def check_lldb_command(self, lldb_command, contains_string, assert_msg):
        response = self.vscode.request_evaluate('`%s' % (lldb_command))
        output = response['body']['result']
        self.assertIn(contains_string, output,
                      ("""Verify %s by checking the command output:\n"""
                       """'''\n%s'''\nfor the string: "%s" """ % (
                       assert_msg, output, contains_string)))

    @skipIfWindows
    @skipIfRemote
    def test_scopes_variables_setVariable_evaluate(self):
        '''
            Tests that the "scopes" request causes the currently selected
            thread and frame to be updated. There are no DAP packets that tell
            lldb-vscode which thread and frame are selected other than the
            "scopes" request. lldb-vscode will now select the thread and frame
            for the latest "scopes" request that it receives.

            The LLDB command interpreter needs to have the right thread and
            frame selected so that commands executed in the debug console act
            on the right scope. This applies both to the expressions that are
            evaluated and the lldb commands that start with the backtick
            character.
        '''
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = 'main.cpp'
        breakpoint1_line = line_number(source, '// breakpoint 1')
        lines = [breakpoint1_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(len(breakpoint_ids), len(lines),
                        "expect correct number of breakpoints")
        self.continue_to_breakpoints(breakpoint_ids)
        # Cause a "scopes" to be sent for frame zero which should update the
        # selected thread and frame to frame 0.
        self.vscode.get_local_variables(frameIndex=0)
        # Verify frame #0 is selected in the command interpreter by running
        # the "frame select" command with no frame index which will print the
        # currently selected frame.
        self.check_lldb_command("frame select", "frame #0",
                                "frame 0 is selected")

        # Cause a "scopes" to be sent for frame one which should update the
        # selected thread and frame to frame 1.
        self.vscode.get_local_variables(frameIndex=1)
        # Verify frame #1 is selected in the command interpreter by running
        # the "frame select" command with no frame index which will print the
        # currently selected frame.

        self.check_lldb_command("frame select", "frame #1",
                                "frame 1 is selected")
