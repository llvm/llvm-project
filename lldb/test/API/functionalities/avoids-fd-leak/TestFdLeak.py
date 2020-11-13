"""
Test whether a process started by lldb has no extra file descriptors open.
"""



import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class AvoidsFdLeakTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    # The check for descriptor leakage needs to be implemented differently
    # here.
    @skipIfWindows
    @skipIfTargetAndroid()  # Android have some other file descriptors open by the shell
    @skipIfDarwinEmbedded # <rdar://problem/33888742>  # debugserver on ios has an extra fd open on launch
    def test_fd_leak_basic(self):
        self.do_test([])

    # The check for descriptor leakage needs to be implemented differently
    # here.
    @skipIfWindows
    @skipIfTargetAndroid()  # Android have some other file descriptors open by the shell
    @skipIfDarwinEmbedded # <rdar://problem/33888742>  # debugserver on ios has an extra fd open on launch
    def test_fd_leak_log(self):
        self.do_test(["log enable -f '/dev/null' lldb commands"])

    def do_test(self, commands):
        self.build()
        exe = self.getBuildArtifact("a.out")

        for c in commands:
            self.runCmd(c)

        target = self.dbg.CreateTarget(exe)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        self.assertTrue(
            process.GetState() == lldb.eStateExited,
            "Process should have exited.")
        self.assertTrue(
            process.GetExitStatus() == 0,
            "Process returned non-zero status. Were incorrect file descriptors passed?")

    # The check for descriptor leakage needs to be implemented differently
    # here.
    @skipIfWindows
    @skipIfTargetAndroid()  # Android have some other file descriptors open by the shell
    @skipIfDarwinEmbedded # <rdar://problem/33888742>  # debugserver on ios has an extra fd open on launch
    def test_fd_leak_multitarget(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', lldb.SBFileSpec("main.c", False))
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        process1 = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process1, PROCESS_IS_VALID)
        self.assertTrue(
            process1.GetState() == lldb.eStateStopped,
            "Process should have been stopped.")

        target2 = self.dbg.CreateTarget(exe)
        process2 = target2.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process2, PROCESS_IS_VALID)

        self.assertTrue(
            process2.GetState() == lldb.eStateExited,
            "Process should have exited.")
        self.assertTrue(
            process2.GetExitStatus() == 0,
            "Process returned non-zero status. Were incorrect file descriptors passed?")
