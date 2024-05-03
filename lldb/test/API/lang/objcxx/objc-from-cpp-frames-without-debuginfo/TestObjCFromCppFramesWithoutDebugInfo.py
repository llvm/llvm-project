"""
Test that the the expression parser enables ObjC support
when stopped in a C++ frame without debug-info.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCFromCppFramesWithoutDebugInfo(TestBase):
    def test(self):
        self.build()
        (_, process, _, _) = lldbutil.run_to_name_breakpoint(self, "main")

        self.assertState(process.GetState(), lldb.eStateStopped)
        self.expect("expr id", error=False)
