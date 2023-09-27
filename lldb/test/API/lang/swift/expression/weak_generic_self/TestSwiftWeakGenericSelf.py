import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftWeakGenericSelf(TestBase):
    @swiftTest
    def test(self):
        """Confirms that expression evaluation works with a generic class
        type within a closure that weakly captures it"""
        self.build()
        _, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect("expression self", substrs=["GenericClass<Int>?", "t = 42"])
