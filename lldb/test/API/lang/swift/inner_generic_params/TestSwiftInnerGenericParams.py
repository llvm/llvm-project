import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftInnerGenericParams(TestBase):
    @requireNotEmbeddedSwift
    @swiftTest
    def test_outermost_params(self):
        """Sanity check: a non-generic method works."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break in method", lldb.SBFileSpec("main.swift")
        )
        self.expect("expression --bind-generic-types false -- value", substrs=["42"])

    @requireNotEmbeddedSwift
    @swiftTest
    def test_inner_params_are_declined(self):
        """A generic method in a generic context is not supported."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break in generic method", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "expression --bind-generic-types false -- value",
            error=True,
            substrs=[
                "Could not evaluate the expression without binding generic types."
            ],
        )
        # The default retries with the generic parameters bound, which works.
        self.expect("expression value", substrs=["42"])
        self.expect("expression --bind-generic-types true -- value", substrs=["42"])
