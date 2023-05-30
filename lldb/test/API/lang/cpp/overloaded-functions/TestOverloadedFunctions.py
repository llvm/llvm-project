"""
Tests that functions with the same name are resolved correctly.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class OverloadedFunctionsTestCase(TestBase):
    def test_with_run_command(self):
        """Test that functions with the same name are resolved correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// breakpoint", lldb.SBFileSpec("main.cpp")
        )

        self.expect("expression -- Dump(myB)", startstr="(int) $0 = 2")

        self.expect("expression -- Static()", startstr="(int) $1 = 1")
