"""
Test that you can set breakpoint and hit the C++ language exception breakpoint
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCPPExceptionBreakpoint (TestBase):
    my_var = 10

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24538")
    def test_cpp_exception_breakpoint(self):
        """Test setting and hitting the C++ exception breakpoint."""
        self.build()
        self.do_cpp_exception_bkpt()

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24538")
    def test_dummy_target_cpp_exception_breakpoint(self):
        """Test setting and hitting the C++ exception breakpoint from dummy target."""
        self.build()
        self.do_dummy_target_cpp_exception_bkpt()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.c"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_cpp_exception_bkpt(self):
        exe = self.getBuildArtifact("a.out")
        error = lldb.SBError()

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        exception_bkpt = self.target.BreakpointCreateForException(
            lldb.eLanguageTypeC_plus_plus, False, True)
        self.assertTrue(
            exception_bkpt.IsValid(),
            "Created exception breakpoint.")

        process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
            process, exception_bkpt)
        self.assertEquals(len(thread_list), 1,
                        "One thread stopped at the exception breakpoint.")

    def do_dummy_target_cpp_exception_bkpt(self):
        exe = self.getBuildArtifact("a.out")
        error = lldb.SBError()

        dummy_exception_bkpt = self.dbg.GetDummyTarget().BreakpointCreateForException(
            lldb.eLanguageTypeC_plus_plus, False, True)
        self.assertTrue(
            dummy_exception_bkpt.IsValid(),
            "Created exception breakpoint in dummy target.")

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        exception_bkpt = self.target.GetBreakpointAtIndex(0)
        self.assertTrue(
            exception_bkpt.IsValid(),
            "Target primed with exception breakpoint from dummy target.")

        process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
           process, exception_bkpt)
        self.assertEquals(len(thread_list), 1,
                       "One thread stopped at the exception breakpoint.")
