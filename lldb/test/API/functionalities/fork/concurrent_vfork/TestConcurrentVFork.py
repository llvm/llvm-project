"""
Make sure that the concurrent vfork() from multiple threads works correctly.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestConcurrentVFork(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def build_run_to_breakpoint(self, use_fork, call_exec):
        self.build()

        args = []
        if use_fork:
            args.append("--fork")
        if call_exec:
            args.append("--exec")
        launch_info = lldb.SBLaunchInfo(args)
        launch_info.SetWorkingDirectory(self.getBuildDir())

        return lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

    def follow_parent_helper(self, use_fork, call_exec):
        (target, process, thread, bkpt) = self.build_run_to_breakpoint(
            use_fork, call_exec
        )

        parent_pid = target.FindFirstGlobalVariable("g_pid").GetValueAsUnsigned()
        self.runCmd("settings set target.process.follow-fork-mode parent")
        self.runCmd("settings set target.process.stop-on-exec False", check=False)
        self.expect(
            "continue", substrs=[f"Process {parent_pid} exited with status = 0"]
        )

    def follow_child_helper(self, use_fork, call_exec):
        self.build_run_to_breakpoint(use_fork, call_exec)

        self.runCmd("settings set target.process.follow-fork-mode child")
        self.runCmd("settings set target.process.stop-on-exec False", check=False)
        # Child process exits with code "index + 10" since index is [0-4]
        # so the exit code should be 1[0-4]
        self.expect("continue", patterns=[r"exited with status = 1[0-4]"])

    @skipUnlessPlatform(["linux"])
    # https://github.com/llvm/llvm-project/issues/85084.
    @skipIf(oslist=["linux"])
    def test_follow_parent_vfork_no_exec(self):
        """
        Make sure that debugging concurrent vfork() from multiple threads won't crash lldb during follow-parent.
        And follow-parent successfully detach all child processes and exit debugger without calling exec.
        """
        self.follow_parent_helper(use_fork=False, call_exec=False)

    @skipUnlessPlatform(["linux"])
    # https://github.com/llvm/llvm-project/issues/85084.
    @skipIf(oslist=["linux"])
    def test_follow_parent_fork_no_exec(self):
        """
        Make sure that debugging concurrent fork() from multiple threads won't crash lldb during follow-parent.
        And follow-parent successfully detach all child processes and exit debugger without calling exec
        """
        self.follow_parent_helper(use_fork=True, call_exec=False)

    @skipUnlessPlatform(["linux"])
    # https://github.com/llvm/llvm-project/issues/85084.
    @skipIf(oslist=["linux"])
    def test_follow_parent_vfork_call_exec(self):
        """
        Make sure that debugging concurrent vfork() from multiple threads won't crash lldb during follow-parent.
        And follow-parent successfully detach all child processes and exit debugger after calling exec.
        """
        self.follow_parent_helper(use_fork=False, call_exec=True)

    @skipUnlessPlatform(["linux"])
    # https://github.com/llvm/llvm-project/issues/85084.
    @skipIf(oslist=["linux"])
    def test_follow_parent_fork_call_exec(self):
        """
        Make sure that debugging concurrent vfork() from multiple threads won't crash lldb during follow-parent.
        And follow-parent successfully detach all child processes and exit debugger after calling exec.
        """
        self.follow_parent_helper(use_fork=True, call_exec=True)

    @skipUnlessPlatform(["linux"])
    # https://github.com/llvm/llvm-project/issues/85084.
    @skipIf(oslist=["linux"])
    def test_follow_child_vfork_no_exec(self):
        """
        Make sure that debugging concurrent vfork() from multiple threads won't crash lldb during follow-child.
        And follow-child successfully detach parent process and exit child process with correct exit code without calling exec.
        """
        self.follow_child_helper(use_fork=False, call_exec=False)

    @skipUnlessPlatform(["linux"])
    # https://github.com/llvm/llvm-project/issues/85084.
    @skipIf(oslist=["linux"])
    def test_follow_child_fork_no_exec(self):
        """
        Make sure that debugging concurrent fork() from multiple threads won't crash lldb during follow-child.
        And follow-child successfully detach parent process and exit child process with correct exit code without calling exec.
        """
        self.follow_child_helper(use_fork=True, call_exec=False)

    @skipUnlessPlatform(["linux"])
    # https://github.com/llvm/llvm-project/issues/85084.
    @skipIf(oslist=["linux"])
    def test_follow_child_vfork_call_exec(self):
        """
        Make sure that debugging concurrent vfork() from multiple threads won't crash lldb during follow-child.
        And follow-child successfully detach parent process and exit child process with correct exit code after calling exec.
        """
        self.follow_child_helper(use_fork=False, call_exec=True)

    @skipUnlessPlatform(["linux"])
    # https://github.com/llvm/llvm-project/issues/85084.
    @skipIf(oslist=["linux"])
    def test_follow_child_fork_call_exec(self):
        """
        Make sure that debugging concurrent fork() from multiple threads won't crash lldb during follow-child.
        And follow-child successfully detach parent process and exit child process with correct exit code after calling exec.
        """
        self.follow_child_helper(use_fork=True, call_exec=True)
