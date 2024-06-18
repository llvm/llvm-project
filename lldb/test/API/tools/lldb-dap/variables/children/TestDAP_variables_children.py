import os

import dap_server
import lldbdap_testcase
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDAP_variables_children(lldbdap_testcase.DAPTestCaseBase):
    def test_get_num_children(self):
        """Test that GetNumChildren is not called for formatters not producing indexed children."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(
            program,
            preRunCommands=[
                "command script import '%s'" % self.getSourcePath("formatter.py")
            ],
        )
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// break here")
        lines = [breakpoint1_line]

        breakpoint_ids = self.set_source_breakpoints(
            source, [line_number(source, "// break here")]
        )
        self.continue_to_breakpoints(breakpoint_ids)

        local_vars = self.dap_server.get_local_variables()
        print(local_vars)
        indexed = next(filter(lambda x: x["name"] == "indexed", local_vars))
        not_indexed = next(filter(lambda x: x["name"] == "not_indexed", local_vars))
        self.assertIn("indexedVariables", indexed)
        self.assertEqual(indexed["indexedVariables"], 1)
        self.assertNotIn("indexedVariables", not_indexed)

        self.assertIn(
            "['Indexed']",
            self.dap_server.request_evaluate(
                "`script formatter.num_children_calls", context="repl"
            )["body"]["result"],
        )
