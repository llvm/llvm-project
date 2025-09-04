"""
Test lldb-dap stack trace containing x86 assembly
"""

import lldbdap_testcase
from lldbsuite.test import lldbplatformutil
from lldbsuite.test.decorators import skipUnlessArch, skipUnlessPlatform
from lldbsuite.test.lldbtest import line_number


class TestDAP_stacktrace_x86(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"] + lldbplatformutil.getDarwinOSTriples())
    def test_stacktrace_x86(self):
        """
        Tests that lldb-dap steps through correctly and the source lines are correct in x86 assembly.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(
            program,
            initCommands=[
                "settings set target.process.thread.step-in-avoid-nodebug false"
            ],
        )

        source = "main.c"
        breakpoint_ids = self.set_source_breakpoints(
            source,
            [line_number(source, "// Break here")],
        )
        self.continue_to_breakpoints(breakpoint_ids)
        self.stepIn()

        frame = self.get_stackFrames()[0]
        self.assertEqual(
            frame["name"],
            "no_branch_func",
            "verify we are in the no_branch_func function",
        )

        self.assertEqual(frame["line"], 1, "verify we are at the start of the function")
        minimum_assembly_lines = (
            line_number(source, "Assembly end")
            - line_number(source, "Assembly start")
            + 1
        )
        self.assertLessEqual(
            10,
            minimum_assembly_lines,
            "verify we have a reasonable number of assembly lines",
        )

        for i in range(2, minimum_assembly_lines):
            self.stepIn()
            frame = self.get_stackFrames()[0]
            self.assertEqual(
                frame["name"],
                "no_branch_func",
                "verify we are still in the no_branch_func function",
            )
            self.assertEqual(
                frame["line"],
                i,
                f"step in should advance a single line in the function to {i}",
            )
