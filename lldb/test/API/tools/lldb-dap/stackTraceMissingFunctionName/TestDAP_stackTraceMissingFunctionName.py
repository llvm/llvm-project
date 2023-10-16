"""
Test lldb-dap stack trace response
"""


import dap
from lldbsuite.test.decorators import *
import os

import lldbdap_testcase
from lldbsuite.test import lldbtest, lldbutil


class TestDAP_stackTraceMissingFunctionName(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    @skipIfRemote
    def test_missingFunctionName(self):
        """
        Test that the stack frame without a function name is given its pc in the response.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        self.continue_to_next_stop()
        frame_without_function_name = self.get_stackFrames()[0]
        self.assertEquals(frame_without_function_name["name"], "0x0000000000000000")
