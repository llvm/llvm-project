"""
Test that the the expression parser enables ObjC support
when stopped in a C++ frame without debug-info.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCFromCppFramesWithoutDebugInfo(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def test(self):
        self.build(debug_info="none")
        (_, process, _, _) = lldbutil.run_to_name_breakpoint(self, "main")

        self.assertState(process.GetState(), lldb.eStateStopped)

        # Tests that we can use builtin Objective-C identifiers.
        self.expect("expr id", error=False)

        # Tests that we can lookup Objective-C decls in the ObjC runtime plugin.
        self.expect_expr(
            "NSString *c; c == nullptr", result_value="true", result_type="bool"
        )
