"""
Test lldb-vscode runInTerminal reverse request
"""


import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import time
import os


class TestVSCode_runInTerminal(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @skipIfRemote
    def test_runInTerminal(self):
        '''
            Tests the "runInTerminal" reverse request. It makes sure that the IDE can
            launch the inferior with the correct environment variables and arguments.
        '''
        program = self.getBuildArtifact("a.out")
        source = 'main.c'
        self.build_and_launch(program, stopOnEntry=True, runInTerminal=True, args=["foobar"], env=["FOO=bar"])
        breakpoint_line = line_number(source, '// breakpoint')

        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        # We verify we actually stopped inside the loop
        counter = int(self.vscode.get_local_variable_value('counter'))
        self.assertTrue(counter > 0)

        # We verify we were able to set the launch arguments
        argc = int(self.vscode.get_local_variable_value('argc'))
        self.assertEqual(argc, 2)

        argv1 = self.vscode.request_evaluate('argv[1]')['body']['result']
        self.assertIn('foobar', argv1)

        # We verify we were able to set the environment
        env = self.vscode.request_evaluate('foo')['body']['result']
        self.assertIn('bar', env)
