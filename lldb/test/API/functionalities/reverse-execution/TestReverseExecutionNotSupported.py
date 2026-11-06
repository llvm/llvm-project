import lldb
import unittest
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestReverseExecutionNotSupported(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_reverse_execution_not_supported(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_name_breakpoint(self, "main")

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
