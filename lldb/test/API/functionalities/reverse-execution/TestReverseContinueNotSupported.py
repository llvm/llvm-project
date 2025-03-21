import lldb
import unittest
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestReverseContinueNotSupported(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_reverse_continue_not_supported(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        main_bkpt = target.BreakpointCreateByName("main", None)
        self.assertTrue(main_bkpt, VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # This will fail gracefully.
        status = process.ContinueInDirection(lldb.eRunReverse)
        self.assertFailure(status)
        # Where gdb-remote is used this starts with "error: gdb-remote" but on Windows it says "error: windows".
        self.assertTrue(
            status.GetCString().endswith(
                " does not support reverse execution of processes"
            )
        )

        self.assertSuccess(process.ContinueInDirection(lldb.eRunForward))
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
