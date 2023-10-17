import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBitfieldIvars(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )

        self.expect_expr(
            "chb->hb->field1", result_type="unsigned int", result_value="0"
        )
        ## This should happen second
        self.expect_expr(
            "chb->hb->field2", result_type="unsigned int", result_value="1"
        )

        self.expect_expr("hb2->field1", result_type="unsigned int", result_value="10")
        self.expect_expr("hb2->field2", result_type="unsigned int", result_value="3")
        self.expect_expr("hb2->field3", result_type="unsigned int", result_value="4")

        self.expect(
            "frame var *hb2",
            substrs=["x =", "100", "field1 =", "10", "field2 =", "3", "field3 =", "4"],
        )

        self.expect_expr(
            "myField",
            result_type="UCBitFields",
            result_children=[
                ValueCheck(name="fieldOne", value="'\\0'"),
                ValueCheck(name="fieldTwo", value="'\\x01'"),
                ValueCheck(name="fieldThree", value="'\\0'"),
                ValueCheck(name="fieldFour", value="'\\0'"),
                ValueCheck(name="fieldFive", value="'\\x01'"),
            ],
        )

    # This test is meant to be xfailed, but running the test triggers an ASan
    # issue, so it must be skipped for now.
    @skipIf
    def testExprWholeObject(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )

        ## FIXME expression with individual bit-fields obtains correct values but not with the whole object
        self.expect(
            "expr *hb2",
            substrs=["x =", "100", "field1 =", "10", "field2 =", "3", "field3 =", "4"],
        )
