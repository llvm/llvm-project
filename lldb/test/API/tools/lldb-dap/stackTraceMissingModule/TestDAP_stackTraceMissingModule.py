"""
Test lldb-dap stack trace when module is missing
"""

from lldbsuite.test.decorators import skipUnlessPlatform
from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase
import re


class TestDAP_stackTraceMissingModule(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessPlatform(["linux"])
    def test_missingModule(self):
        """
        Test that the stack frame without a module still has assembly source.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, commandEscapePrefix="")

        source = "main.c"
        self.set_source_breakpoints(
            source,
            [line_number(source, "// Break here")],
        )
        self.continue_to_next_stop()

        # Evaluate expr -- func
        expr_result = self.dap_server.request_evaluate(
            expression="expr -f pointer -- func",
            context="repl",
        )

        expr_result_address = re.search(
            r"0x[0-9a-fA-F]+", expr_result["body"]["result"]
        )
        self.assertIsNotNone(
            expr_result_address, "Failed to get address of dynamic allocated func"
        )
        func_address = expr_result_address.group(0)

        self.dap_server.request_evaluate(
            expression=f"breakpoint set --address {func_address}",
            context="repl",
        )

        self.continue_to_next_stop()

        frame_without_module = self.get_stackFrames()[0]

        self.assertIn("line", frame_without_module, "Line number missing.")
        self.assertIn("column", frame_without_module, "Column number missing.")
        self.assertIn("source", frame_without_module, "Source location missing.")
        source = frame_without_module["source"]
        self.assertIn("sourceReference", source, "Source reference missing.")
