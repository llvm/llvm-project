"""
Test that argdumper is a viable launching strategy.
"""
import os


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LaunchWithShellExpandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(
        oslist=[
            "windows",
            "linux",
            "freebsd"],
        bugnumber="llvm.org/pr24778 llvm.org/pr22627 llvm.org/pr48349")
    @skipIfDarwinEmbedded # iOS etc don't launch the binary via a shell, so arg expansion won't happen
    @expectedFailureNetBSD
    def test(self):
        self.build()
        target = self.createTestTarget()

        # Create any breakpoints we need
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec("main.cpp", False))
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Ensure we do the expansion with /bin/sh on POSIX.
        os.environ["SHELL"] = '/bin/sh'

        self.runCmd(
            "process launch -X true -w %s -- fi*.tx? () > <" %
            (self.getSourceDir()))

        process = self.process()

        self.assertState(process.GetState(), lldb.eStateStopped,
                         STOPPED_DUE_TO_BREAKPOINT)

        thread = process.GetThreadAtIndex(0)

        self.assertTrue(thread.IsValid(),
                        "Process stopped at 'main' should have a valid thread")

        stop_reason = thread.GetStopReason()

        self.assertEqual(
            stop_reason, lldb.eStopReasonBreakpoint,
            "Thread in process stopped in 'main' should have a stop reason of eStopReasonBreakpoint")

        self.expect_var_path("argv[1]", summary='"file1.txt"')
        self.expect_var_path("argv[2]", summary='"file2.txt"')
        self.expect_var_path("argv[3]", summary='"file3.txt"')
        self.expect_var_path("argv[4]", summary='"file4.txy"')
        self.expect_var_path("argv[5]", summary='"()"')
        self.expect_var_path("argv[6]", summary='">"')
        self.expect_var_path("argv[7]", summary='"<"')
        self.expect_var_path("argc", value='8')

        self.runCmd("process kill")

        self.runCmd(
            'process launch -X true -w %s -- "foo bar"' %
            (self.getSourceDir()))

        process = self.process()

        self.assertState(process.GetState(), lldb.eStateStopped,
                         STOPPED_DUE_TO_BREAKPOINT)

        thread = process.GetThreadAtIndex(0)

        self.assertTrue(thread.IsValid(),
                        "Process stopped at 'main' should have a valid thread")

        stop_reason = thread.GetStopReason()

        self.assertEqual(
            stop_reason, lldb.eStopReasonBreakpoint,
            "Thread in process stopped in 'main' should have a stop reason of eStopReasonBreakpoint")

        self.expect("frame variable argv[1]", substrs=['foo bar'])

        self.runCmd("process kill")

        self.runCmd('process launch -X true -w %s -- foo\ bar'
                    % (self.getBuildDir()))

        process = self.process()

        self.assertState(process.GetState(), lldb.eStateStopped,
                         STOPPED_DUE_TO_BREAKPOINT)

        thread = process.GetThreadAtIndex(0)

        self.assertTrue(thread.IsValid(),
                        "Process stopped at 'main' should have a valid thread")

        stop_reason = thread.GetStopReason()

        self.assertEqual(
            stop_reason, lldb.eStopReasonBreakpoint,
            "Thread in process stopped in 'main' should have a stop reason of eStopReasonBreakpoint")

        self.expect("frame variable argv[1]", substrs=['foo bar'])
