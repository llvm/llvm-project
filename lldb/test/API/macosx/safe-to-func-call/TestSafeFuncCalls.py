"""Test function call thread safety."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSafeFuncCalls(TestBase):

    @skipUnlessDarwin
    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test function call thread safety."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.main_source_spec = lldb.SBFileSpec("main.c")
        break1 = target.BreakpointCreateByName("stopper", 'a.out')
        self.assertTrue(break1, VALID_BREAKPOINT)
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, break1)
        self.assertEqual(len(threads), 1, "Failed to stop at breakpoint 1.")

        self.assertEqual(
            process.GetNumThreads(), 2,
            "Check that the process has two threads when sitting at the stopper() breakpoint")

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
            "Got both expected threads")

        self.assertTrue(main_thread.SafeToCallFunctions(),
                        "It is safe to call functions on the main thread")
        self.assertEqual(
            select_thread.SafeToCallFunctions(), False,
            "It is not safe to call functions on the select thread")
