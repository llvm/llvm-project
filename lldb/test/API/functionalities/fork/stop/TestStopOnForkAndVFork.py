"""
Make sure that we stop on fork and follow mode works.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import os


class TestForkResumesChild(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def do_test(self, mode):
        self.build()
        exe = self.getBuildArtifact("a.out")

        (target, process, _, _) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )
        self.runCmd("settings set target.process.stop-on-fork true")
        self.runCmd(f"settings set target.process.follow-fork-mode {mode}")

        process.Continue()
        self.assertState(
            process.GetState(), lldb.eStateStopped, f"Process should be stopped at fork"
        )
        threads = lldbutil.get_stopped_threads(process, lldb.eStopReasonFork)
        self.assertEqual(len(threads), 1, f"We got a thread stopped for fork.")

        self.expect(
            "continue",
            substrs=[f"exited with status = {0 if mode == 'parent' else 47}"],
        )

    @skipIfWindows
    def test_stop_on_fork_and_follow_parent(self):
        self.do_test("parent")

    @skipIfWindows
    def test_stop_on_fork_and_follow_child(self):
        self.do_test("child")
