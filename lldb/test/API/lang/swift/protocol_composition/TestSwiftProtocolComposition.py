import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftProtocolComposition(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test that expression evaluation can call functions in protocol composition existentials"""

        self.build()
        filespec = lldb.SBFileSpec("main.swift")
        target, process, thread, breakpoint1 = lldbutil.run_to_source_breakpoint(
            self, "break here", filespec
        )
        self.expect("expr c.foo()", substrs=["In class foo"])
        self.expect("expr c.bar()", substrs=["In class bar"])

        self.expect("expr s.foo()", substrs=["In struct foo"])
        self.expect("expr s.bar()", substrs=["In struct bar"])
