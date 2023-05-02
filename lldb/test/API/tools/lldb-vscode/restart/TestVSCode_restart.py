"""
Test lldb-vscode RestartRequest.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
import lldbvscode_testcase


class TestVSCode_restart(lldbvscode_testcase.VSCodeTestCaseBase):

    @skipIfWindows
    @skipIfRemote
    def test_basic_functionality(self):
        '''
            Tests the basic restarting functionality: set two breakpoints in
            sequence, restart at the second, check that we hit the first one.
        '''
        line_A = line_number('main.c', '// breakpoint A')
        line_B = line_number('main.c', '// breakpoint B')

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        [bp_A, bp_B] = self.set_source_breakpoints('main.c', [line_A, line_B])

        # Verify we hit A, then B.
        self.vscode.request_configurationDone()
        self.verify_breakpoint_hit([bp_A])
        self.vscode.request_continue()
        self.verify_breakpoint_hit([bp_B])

        # Make sure i has been modified from its initial value of 0.
        self.assertEquals(int(self.vscode.get_local_variable_value('i')),
                          1234, 'i != 1234 after hitting breakpoint B')

        # Restart then check we stop back at A and program state has been reset.
        self.vscode.request_restart()
        self.verify_breakpoint_hit([bp_A])
        self.assertEquals(int(self.vscode.get_local_variable_value('i')),
                          0, 'i != 0 after hitting breakpoint A on restart')


    @skipIfWindows
    @skipIfRemote
    def test_stopOnEntry(self):
        '''
            Check that the stopOnEntry setting is still honored after a restart.
        '''
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        [bp_main] = self.set_function_breakpoints(['main'])
        self.vscode.request_configurationDone()

        # Once the "configuration done" event is sent, we should get a stopped
        # event immediately because of stopOnEntry.
        stopped_events = self.vscode.wait_for_stopped()
        for stopped_event in stopped_events:
            if 'body' in stopped_event:
                body = stopped_event['body']
                if 'reason' in body:
                    reason = body['reason']
                    self.assertNotEqual(
                        reason, 'breakpoint',
                        'verify stop isn\'t "main" breakpoint')

        # Then, if we continue, we should hit the breakpoint at main.
        self.vscode.request_continue()
        self.verify_breakpoint_hit([bp_main])

        # Restart and check that we still get a stopped event before reaching
        # main.
        self.vscode.request_restart()
        stopped_events = self.vscode.wait_for_stopped()
        for stopped_event in stopped_events:
            if 'body' in stopped_event:
                body = stopped_event['body']
                if 'reason' in body:
                    reason = body['reason']
                    self.assertNotEqual(
                        reason, 'breakpoint',
                        'verify stop after restart isn\'t "main" breakpoint')

