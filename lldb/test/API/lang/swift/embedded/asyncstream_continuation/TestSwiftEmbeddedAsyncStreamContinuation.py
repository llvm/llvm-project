import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedAsyncStreamContinuation(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test frame variable on an AsyncStream continuation in embedded Swift."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "frame variable continuation",
            substrs=[
                "AsyncStream<a.DataItem>.Continuation",
            ],
        )
