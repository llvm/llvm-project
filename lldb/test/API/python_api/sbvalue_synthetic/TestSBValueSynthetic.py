import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSBValueSynthetic(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_str(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.runCmd("command script import formatter.py")
        self.runCmd(
            "type synthetic add --python-class formatter.FooSyntheticProvider Foo"
        )

        formatted = self.frame().FindVariable("foo")
        has_formatted = self.frame().FindVariable("has_foo")
        self.expect(str(formatted), exe=False, substrs=["synth_child"])
        self.expect(str(has_formatted), exe=False, substrs=["synth_child"])

    def test_synth_arr(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        point_arr = self.frame().FindVariable("point_arr")
        point_ptr = self.frame().FindVariable("point_ptr")
        for v in [point_arr, point_ptr]:
            for i in range(3):
                child = v.GetChildAtIndex(i, lldb.eDynamicDontRunTarget, True)
                check = ValueCheck(
                    name=f"[{i}]",
                    type="Point",
                    children=[
                        ValueCheck(name="x", value=str(2 * i + 1)),
                        ValueCheck(name="y", value=str(2 * i + 2)),
                    ],
                )
                check.check_value(self, child, f"{child}, child {i} of {v.GetName()}")

        int_arr = self.frame().FindVariable("int_arr")
        int_ptr = self.frame().FindVariable("int_ptr")
        for v in [int_arr, int_ptr]:
            for i in range(3):
                child = v.GetChildAtIndex(i, lldb.eDynamicDontRunTarget, True)
                check = ValueCheck(name=f"[{i}]", type="int", value=str(i + 1))
                check.check_value(self, child, f"{child}, child {i} of {v.GetName()}")
