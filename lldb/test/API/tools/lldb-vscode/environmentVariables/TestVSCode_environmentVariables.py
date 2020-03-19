"""
Test lldb-vscode environment variables
"""


import lldbvscode_testcase
import unittest2
import vscode
import os
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class TestVSCode_variables(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def getEnvOutputByProgram(self):
        env = {}
        for line in self.get_stdout().encode('utf-8').splitlines():
            (name, value) = line.split("=")
            env[name] = value
        return env

    @skipIfWindows
    @skipIfRemote
    def test_empty_environment(self):
        """
            Tests running a process with an empty environment
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        self.continue_to_exit()
        env = self.getEnvOutputByProgram()

        self.assertNotIn("PATH", env)

    @skipIfWindows
    @skipIfRemote
    def test_inheriting_environment(self):
        """
            Tests running a process inheriting the environment
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, inheritEnvironment=True, env=["FOO=bar"])
        self.continue_to_exit()
        env = self.getEnvOutputByProgram()

        self.assertEqual(env["PATH"], os.environ['PATH'])
        self.assertEqual(env["FOO"], "bar")

    @skipIfWindows
    @skipIfRemote
    def test_override_when_inheriting_environment(self):
        """
            Tests the environment variables priority.
            The launch.json's environment has precedence.
        """
        program = self.getBuildArtifact("a.out")
        new_path_value = "#" + os.environ["PATH"]

        self.build_and_launch(
            program,
            inheritEnvironment=True,
            env=["PATH=" + new_path_value])
        self.continue_to_exit()
        env = self.getEnvOutputByProgram()

        self.assertEqual(env["PATH"], new_path_value)

    @skipIfWindows
    @skipIfRemote
    def test_empty_environment_custom_launcher(self):
        """
            Tests running a process with an empty environment from a custom
            launcher
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_create_debug_adaptor()

        launchCommands = [
            'target create "%s"' % (program),
            "run"
        ]
        self.launch(launchCommands=launchCommands)
        self.continue_to_exit()
        env = self.getEnvOutputByProgram() 
        self.assertNotIn("PATH", env)

    @skipIfWindows
    @skipIfRemote
    def test_inheriting_environment_custom_launcher(self):
        """
            Tests running a process from a custom launcher inheriting the
            environment
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_create_debug_adaptor()

        launchCommands = [
            'target create "%s"' % (program),
            "run"
        ]
        self.launch(launchCommands=launchCommands, inheritEnvironment=True)
        self.continue_to_exit()
        env = self.getEnvOutputByProgram() 
        self.assertIn("PATH", env)
