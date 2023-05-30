"""
Test that breakpoints correctly work in an thumb function in an arbitrary
named codesection.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBreakpointThumbCodesection(TestBase):
    @skipIf(archs=no_match(["arm"]))
    @skipIf(archs=["arm64"])
    @skipIfDarwinEmbedded  # codegen on darwin always defaults to thumb for armv7/armv7k targets
    def test_breakpoint(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        line = line_number("main.c", "// Set break point at this line.")

        self.runCmd("target create %s" % exe)
        bpid = lldbutil.run_break_set_by_file_and_line(self, "main.c", line)

        self.runCmd("run")

        self.assertIsNotNone(
            lldbutil.get_one_thread_stopped_at_breakpoint_id(self.process(), bpid),
            "Process is not stopped at breakpoint",
        )

        self.process().Continue()
        self.assertState(self.process().GetState(), lldb.eStateExited, PROCESS_EXITED)
