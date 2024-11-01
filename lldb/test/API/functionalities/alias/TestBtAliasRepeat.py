import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return", lldb.SBFileSpec("main.c"))

        # Expect "frame #0" but not "frame #1".
        self.expect("bt 1", inHistory=True, patterns=["frame #0", "^(?!.*frame #1)"])

        # Run an empty command to run the repeat command for `bt`.
        # The repeat command for `bt N` lists the subsequent N frames.
        #
        # In this case, after printing the frame 0 with `bt 1`, the repeat
        # command will print "frame #1" (and won't print "frame #0").
        self.expect("", patterns=["^(?!.*frame #0)", "frame #1"])
