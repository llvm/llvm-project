import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftEmbeddedExpression(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect("expr a.foo()", substrs=["(Int)", " = 16"])
