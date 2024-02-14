"""
Make sure that the concurrent vfork() from multiple threads works correctly.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestConcurrentVFork(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def get_pid_from_variable(self):
        target = self.dbg.GetTargetAtIndex(0)
        return target.FindFirstGlobalVariable("g_pid").GetValueAsUnsigned()

    @skipIfWindows
    def test_vfork_follow_parent(self):
        """
        Make sure that debugging concurrent vfork() from multiple threads won't crash lldb during follow-parent.
        And follow-parent successfully detach all child processes and exit debugger.
        """

        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        parent_pid = self.get_pid_from_variable()
        self.runCmd("settings set target.process.follow-fork-mode parent")
        self.expect(
            "continue", substrs=[f"Process {parent_pid} exited with status = 0"]
        )

    @skipIfWindows
    def test_vfork_follow_child(self):
        """
        Make sure that debugging concurrent vfork() from multiple threads won't crash lldb during follow-child.
        And follow-child successfully detach parent process and exit child process with correct exit code.
        """
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        self.runCmd("settings set target.process.follow-fork-mode child")
        # Child process exits with code "index + 10" since index is [0-4]
        # so the exit code should be 1[0-4]
        self.expect("continue", patterns=[r"exited with status = 1[0-4]"])
