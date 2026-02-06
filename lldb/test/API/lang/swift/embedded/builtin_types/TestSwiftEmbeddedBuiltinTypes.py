import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedBuiltinTypes(TestBase):
    """
    Tests for builtin type descriptors in embedded Swift.
    """

    @skipUnlessDarwin
    @swiftTest
    def test_without_ast(self):
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable holder",
            substrs=["ClosureHolder", "callback", "doNothing"],
        )

