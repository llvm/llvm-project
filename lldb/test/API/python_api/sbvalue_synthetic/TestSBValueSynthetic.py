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
