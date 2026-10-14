import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SyntheticOneLineSummariesTestCase(TestBase):
    def test(self):
        """Test that the presence of a synthetic child provider doesn't prevent one-line-summaries."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))

        # set up the synthetic children provider
        self.runCmd("script from MyStringFormatter import *")
        self.runCmd("type synth add -l MyStringSynthProvider MyString")
        self.runCmd('type summary add --summary-string "${var.guts}" MyString')

        self.expect(
            "frame variable s",
            substrs=['a = "hello", b = "world"'],
        )
