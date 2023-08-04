import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftFunctionDefinition(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    def test(self):
        """Test that persistent variables are mutable."""
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        self.expect("expr struct $S { let v : Int }")
        self.expect("expr func $dup<T>(_ t: T) -> (T, T) { return (t, t) }")
        self.expect("expr $dup($S(v: 1))", substrs=["($S, $S)", "v = 1", "v = 1"])
