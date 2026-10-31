"""Test function call thread safety."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSafeFuncCalls(TestBase):
    @requireDarwin
    @add_test_categories(["pyapi"])
    def test_with_python_api(self):
        """Test function call thread safety."""
        self.build()
        _, process, _, _ = lldbutil.run_to_name_breakpoint(
            self, "stopper", bkpt_module="a.out"
        )

        self.assertEqual(
            process.GetNumThreads(),
            2,
            "Check that the process has two threads when sitting at the stopper() breakpoint",
        )

        main_thread = lldb.SBThread()
        select_thread = lldb.SBThread()
        for idx in range(0, process.GetNumThreads()):
            t = process.GetThreadAtIndex(idx)
            if t.GetName() == "main thread":
                main_thread = t
            if t.GetName() == "select thread":
                select_thread = t

        self.assertTrue(
            main_thread.IsValid() and select_thread.IsValid(),
            "Got both expected threads",
        )

        self.assertTrue(
            main_thread.SafeToCallFunctions(),
            "It is safe to call functions on the main thread",
        )
        self.assertFalse(
            select_thread.SafeToCallFunctions(),
            "It is not safe to call functions on the select thread",
        )
