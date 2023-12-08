import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        self.dbg.HandleCommand(
            f"type summary add --expand -s 'some summary' SummaryAndChildren"
        )
        self.dbg.HandleCommand(f"type summary add -s 'some summary' SummaryOnly")

        self.expect(
            "v summary_and_children_ref", substrs=["some summary", "child = 30"]
        )
        self.expect(
            "v summary_only_ref", patterns=["some summary", "(?s)^(?!.*child = )"]
        )
        self.expect(
            "v children_only_ref", patterns=["(?s)^(?!.*some summary)", "child = 30"]
        )
