"""
Test the error messages emitted by invalid 'watchpoint set' invocations.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchpointSetErrorsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_set_without_subcommand(self):
        """'watchpoint set' without a subcommand prints its help."""
        self.build_and_run()
        self.expect(
            "watchpoint set",
            substrs=[
                "Commands for setting a watchpoint.",
                "The following subcommands are supported:",
                "Set a watchpoint on an address by supplying an expression.",
                "Set a watchpoint on a variable.",
            ],
        )

    def test_variable_without_argument(self):
        """'watchpoint set variable' requires a variable name."""
        self.build_and_run()
        self.expect(
            "watchpoint set variable -w read_write",
            error=True,
            substrs=["error: required argument missing"],
        )

    def test_expression_without_argument(self):
        """'watchpoint set expression' requires an expression."""
        self.build_and_run()
        self.expect(
            "watchpoint set expression -w write --",
            error=True,
            substrs=["error: expression evaluation of address to watch failed"],
        )

    def test_expression_not_an_address(self):
        """'watchpoint set expression' rejects a non-address expression."""
        self.build_and_run()
        self.expect(
            "watchpoint set expression MyAggregateDataType",
            error=True,
            substrs=["error: expression did not evaluate to an address"],
        )

    def test_negative_size(self):
        """'watchpoint set' rejects a negative --size value."""
        self.build_and_run()
        self.expect(
            "watchpoint set variable -s -128",
            error=True,
            substrs=["error: invalid --size option value"],
        )
