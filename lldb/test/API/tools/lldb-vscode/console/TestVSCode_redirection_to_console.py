import unittest2
import vscode
import json
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase


class TestVSCode_redirection_to_console(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows
    @skipIfRemote
    def test(self):
        """
            Without proper stderr and stdout redirection, the following code would throw an
            exception, like the following:

                Exception: unexpected malformed message from lldb-vscode
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(
            program,
            lldbVSCodeEnv={"LLDB_VSCODE_TEST_STDOUT_STDERR_REDIRECTION": ""})

        source = 'main.cpp'

        breakpoint1_line = line_number(source, '// breakpoint 1')
        breakpoint_ids = self.set_source_breakpoints(source, [breakpoint1_line])

        self.assertEqual(len(breakpoint_ids), 1,
                        "expect correct number of breakpoints")
        self.continue_to_breakpoints(breakpoint_ids)

        self.assertIn('argc', json.dumps(self.vscode.get_local_variables(frameIndex=1)))
