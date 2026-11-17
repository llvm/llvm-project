"""Test that SBFrame::FindValue finds things but does not duplicate the entire variables list"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SBFrameFindValueTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_formatters_api(self):
        """Test that SBFrame::FindValue finds things but does not duplicate the entire variables list"""
        self.build()
        self.setTearDownCleanup()

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main.cpp")
        )
        self.thread = thread
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        self.assertEqual(
            self.frame.GetVariables(True, True, False, True).GetSize(),
            2,
            "variable count is off",
        )
        self.assertFalse(
            self.frame.FindValue(
                "NoSuchThing",
                lldb.eValueTypeVariableArgument,
                lldb.eDynamicCanRunTarget,
            ).IsValid(),
            "found something that should not be here",
        )
        self.assertEqual(
            self.frame.GetVariables(True, True, False, True).GetSize(),
            2,
            "variable count is off after failed FindValue()",
        )
        self.assertTrue(
            self.frame.FindValue(
                "a", lldb.eValueTypeVariableArgument, lldb.eDynamicCanRunTarget
            ).IsValid(),
            "FindValue() didn't find an argument",
        )
        self.assertEqual(
            self.frame.GetVariables(True, True, False, True).GetSize(),
            2,
            "variable count is off after successful FindValue()",
        )
