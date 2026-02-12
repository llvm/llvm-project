import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedEmptyArray(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test empty array formatting in embedded Swift."""
        self.build()
        # Disable the ast context so we don't pass the test by silently falling 
        # back to the ast context implementation.
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable emptyInts",
            substrs=["0 values"],
        )

        self.expect(
            "frame variable emptyPoints",
            substrs=["0 values"],
        )

        self.expect(
            "frame variable nonEmptyInts",
            substrs=["3 values"],
        )
