import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import re


class DynamicValueTemplates(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        intf_check = ValueCheck(
            type="Interface", children=[ValueCheck(name="intf", value="321")]
        )
        t1_children = [intf_check, ValueCheck(name="templ", value="1")]

        self.expect_var_path("t1_int", type="Template1<int> *", children=t1_children)
        self.expect_var_path(
            "t1_single",
            type=re.compile(r"^Template1<Single<int> ?> \*$"),
            children=t1_children,
        )
        self.expect_var_path(
            "t1_singleception",
            type=re.compile(r"^Template1<Single<Single<int> ?> ?> \*$"),
            children=t1_children,
        )
        self.expect_var_path(
            "t1_double",
            type=re.compile(r"^Template1<Double<int, float> ?> \*$"),
            children=t1_children,
        )

        t2_children = [intf_check, ValueCheck(name="templ", value="2")]
        self.expect_var_path(
            "t2_int_bool", type="Template2<int, bool> *", children=t2_children
        )
        self.expect_var_path(
            "t2_single_double",
            type=re.compile(r"^Template2<Single<int>, Double<int, int> ?> \*$"),
            children=t2_children,
        )
        self.expect_var_path(
            "t2_complex",
            type=re.compile(
                r"^Template2<Single<int>, Double<Single<bool>, Single<char> ?> ?> \*$"
            ),
            children=t2_children,
        )

        self.expect_var_path(
            "type_and_const",
            type="TypeAndConstant<int, 42> *",
            children=[
                intf_check,
                ValueCheck(name="templ", value="3"),
            ],
        )
