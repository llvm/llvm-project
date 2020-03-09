"""
Test that the target.process.thread.max-backtrace-depth setting works.
"""

import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BacktraceLimitSettingTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_backtrace_depth(self):
        """Test that the max-backtrace-depth setting limits backtraces."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                "Set a breakpoint here", self.main_source_file)
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("settings set target.process.thread.max-backtrace-depth 30", result)
        self.assertEqual(True, result.Succeeded())
        self.assertEqual(30, thread.GetNumFrames())
