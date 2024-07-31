"""
Test lldb-dap repl mode detection
"""

import lldbdap_testcase
import dap_server
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDAP_repl_mode_detection(lldbdap_testcase.DAPTestCaseBase):
    def assertEvaluate(self, expression, regex):
        self.assertRegex(
            self.dap_server.request_evaluate(expression, context="repl")["body"][
                "result"
            ],
            regex,
        )

    def test_completions(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        breakpoint2_line = line_number(source, "// breakpoint 2")

        self.set_source_breakpoints(source, [breakpoint1_line, breakpoint2_line])

        self.assertEvaluate(
            "`command regex user_command s/^$/platform/", r"\(lldb\) command regex"
        )
        self.assertEvaluate(
            "`command alias alias_command platform", r"\(lldb\) command alias"
        )
        self.assertEvaluate(
            "`command alias alias_command_with_arg platform select --sysroot %1 remote-linux",
            r"\(lldb\) command alias",
        )

        self.continue_to_next_stop()
        self.assertEvaluate("user_command", "474747")
        self.assertEvaluate("alias_command", "474747")
        self.assertEvaluate("alias_command_with_arg", "474747")
        self.assertEvaluate("platform", "474747")

        self.continue_to_next_stop()
        platform_help_needle = "Commands to manage and create platforms"
        self.assertEvaluate("user_command", platform_help_needle)
        self.assertEvaluate("alias_command", platform_help_needle)
        self.assertEvaluate(
            "alias_command_with_arg " + self.getBuildDir(), "Platform: remote-linux"
        )
        self.assertEvaluate("platform", platform_help_needle)
