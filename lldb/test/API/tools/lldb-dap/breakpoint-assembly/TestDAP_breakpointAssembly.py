"""
Test lldb-dap setBreakpoints request
"""


import dap_server
import shutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


class TestDAP_setBreakpointsAssembly(lldbdap_testcase.DAPTestCaseBase):
    # @skipIfWindows
    def test_functionality(self):
        """Tests hitting assembly source breakpoints"""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        self.dap_server.request_evaluate(
            "`settings set stop-disassembly-display no-debuginfo", context="repl"
        )

        assmebly_func_breakpoints = self.set_function_breakpoints(["assembly_func"])
        self.continue_to_breakpoints(assmebly_func_breakpoints)

        assembly_func_frame = self.get_stackFrames()[0]
        self.assertIn(
            "sourceReference",
            assembly_func_frame.get("source"),
            "Expected assembly source frame",
        )

        line = assembly_func_frame["line"]

        # Set an assembly breakpoint in the next line and check that it's hit
        assembly_breakpoint_ids = self.set_source_breakpoints_assembly(
            assembly_func_frame["source"]["sourceReference"], [line + 1]
        )
        self.continue_to_breakpoints(assembly_breakpoint_ids)
