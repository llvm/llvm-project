import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedTypealias(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect("frame variable simple", substrs=["x = 42"])

        self.expect("frame variable multi", substrs=["i = 1", "d = 2.5", "b = true"])

        self.expect("frame variable optionalNil", substrs=["maybeInt = nil"])
        self.expect("frame variable optionalSome", substrs=["maybeInt = 99"])

        self.expect("frame variable tupleStruct.pair", substrs=["0 = 10", "1 = 20"])

        self.expect("frame variable nestedAlias", substrs=["nested = 123"])

        self.expect("frame variable genericAlias", substrs=["value = 314", "aliased = 100"])

        self.expect("frame variable classAlias", substrs=["x = 500"])

        self.expect("frame variable outer", substrs=["x = 777"])
