"""
Test that breakpoints in an IT instruction don't fire if their condition is
false.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBreakpointIt(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(archs=no_match(["arm"]))
    @skipIf(archs=["arm64", "arm64e", "arm64_32"])
    def test_false(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.runCmd("target create %s" % exe)
        lldbutil.run_break_set_by_symbol(self, "bkpt_false",
                extra_options="--skip-prologue 0")

        self.runCmd("run")
        self.assertState(self.process().GetState(), lldb.eStateExited,
                "Breakpoint does not get hit")

    @skipIf(archs=no_match(["arm"]))
    @skipIf(archs=["arm64", "arm64e", "arm64_32"])
    def test_true(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.runCmd("target create %s" % exe)
        bpid = lldbutil.run_break_set_by_symbol(self, "bkpt_true",
                extra_options="--skip-prologue 0")

        self.runCmd("run")
        self.assertIsNotNone(lldbutil.get_one_thread_stopped_at_breakpoint_id(
            self.process(), bpid))
