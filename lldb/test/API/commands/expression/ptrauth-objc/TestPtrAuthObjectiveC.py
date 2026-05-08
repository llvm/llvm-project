import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbarm64e import Arm64eTestBase


class TestPtrAuthObjectiveC(Arm64eTestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def test_objc_message_send(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m", False)
        )

        self.expect_expr(
            "[obj doubleValue]",
            result_type="int",
            result_value="42",
        )

    def test_objc_message_send_with_arg(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m", False)
        )

        self.expect_expr(
            "[obj addValue:9]",
            result_type="int",
            result_value="30",
        )

    def test_objc_alloc_and_message(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m", False)
        )

        self.expect_expr(
            "PtrAuthTestObj *tmp = (PtrAuthTestObj *)[[PtrAuthTestObj alloc] init]; "
            "tmp.value = 7; [tmp doubleValue]",
            result_type="int",
            result_value="14",
        )

    def test_objc_derived_class(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m", False)
        )

        self.expect_expr(
            "[derived tripleValue]",
            result_type="int",
            result_value="30",
        )

        self.expect_expr(
            "[derived doubleValue]",
            result_type="int",
            result_value="20",
        )

    def test_objc_isa_check(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m", False)
        )

        self.expect_expr(
            "(bool)[derived isKindOfClass:[PtrAuthTestObj class]]",
            result_type="bool",
            result_value="true",
        )

        self.expect_expr(
            "(bool)[obj isKindOfClass:[PtrAuthDerived class]]",
            result_type="bool",
            result_value="false",
        )
