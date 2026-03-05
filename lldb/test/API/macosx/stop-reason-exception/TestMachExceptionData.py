"""
Test that we get the type code and subcode for MachExceptions
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestMachExceptionData(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_exc_bad_access(self):
        """Test that we get type 1, code 1 and the right address for
        a EXC_BAD_ACCESS mach exception."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        # Now continue and we should crash:
        process.Continue()
        self.assertEqual(
            lldb.eStopReasonException,
            thread.GetStopReason(),
            "Got the right stop reason",
        )
        self.assertEqual(thread.GetStopReasonDataCount(), 3, "Got all the codes")
        self.assertEqual(thread.stop_reason_data[0], 1, "1 is EXC_BAD_ACCESS")
        self.assertEqual(thread.stop_reason_data[1], 1, "1 is 'access invalid memory'")
        self.assertEqual(thread.stop_reason_data[2], 0x400, "That's the bad address")
