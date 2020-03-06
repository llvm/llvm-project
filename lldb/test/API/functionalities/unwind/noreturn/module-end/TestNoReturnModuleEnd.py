"""
Test that we properly display the backtrace when a noreturn function happens to
be at the end of the stack.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestNoreturnModuleEnd(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        super(TestNoreturnModuleEnd, self).setUp()
        self._initial_platform = lldb.DBG.GetSelectedPlatform()

    def tearDown(self):
        lldb.DBG.SetSelectedPlatform(self._initial_platform)
        super(TestNoreturnModuleEnd, self).tearDown()

    def test(self):
        target = self.dbg.CreateTarget("test.out")
        process = target.LoadCore("test.core")
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 1)

        thread = process.GetSelectedThread()
        self.assertTrue(thread.IsValid())

        backtrace = [
            ["func2", 3],
            ["func1", 8],
            ["_start", 8],
        ]
        self.assertEqual(thread.GetNumFrames(), len(backtrace))
        for i in range(len(backtrace)):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame.IsValid())
            symbol = frame.GetSymbol()
            self.assertTrue(symbol.IsValid())
            self.assertEqual(symbol.GetName(), backtrace[i][0])
            function_start = symbol.GetStartAddress().GetLoadAddress(target)
            self.assertEquals(function_start + backtrace[i][1], frame.GetPC())

        self.dbg.DeleteTarget(target)
