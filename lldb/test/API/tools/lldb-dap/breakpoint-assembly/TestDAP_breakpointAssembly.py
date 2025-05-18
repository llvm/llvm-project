"""
Test lldb-dap setBreakpoints request
"""


import dap_server
import shutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
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

        finish_line = line_number("main.c", "// Break here")
        finish_breakpoints = self.set_source_breakpoints("main.c", [finish_line])

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
        source_reference = assembly_func_frame["source"]["sourceReference"]
        assembly_breakpoint_ids = self.set_source_breakpoints_assembly(
            source_reference, [line + 1]
        )
        self.continue_to_breakpoints(assembly_breakpoint_ids)

        # Continue again and verify it hits in the next function call
        self.continue_to_breakpoints(assmebly_func_breakpoints)
        self.continue_to_breakpoints(assembly_breakpoint_ids)

        # Clear the breakpoint and then check that the assembly breakpoint does not hit next time
        self.set_source_breakpoints_assembly(source_reference, [])
        self.continue_to_breakpoints(assmebly_func_breakpoints)
        self.continue_to_breakpoints(finish_breakpoints)
