"""
Test that we correctly handle inline namespaces.
"""

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInlineNamespace(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        # The 'A::B::f' function must be found via 'A::f' as 'B' is an inline
        # namespace.
        self.expect("expr A::f()", substrs=['$0 = 3'])
        # But we should still find the function when we pretend the inline
        # namespace is not inline.
        self.expect("expr A::B::f()", substrs=['$1 = 3'])
