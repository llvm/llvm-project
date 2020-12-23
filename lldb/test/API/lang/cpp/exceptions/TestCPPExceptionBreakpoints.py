"""
Test lldb exception breakpoint command for CPP.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CPPBreakpointTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.source = 'exceptions.cpp'
        self.catch_line = line_number(
            self.source, '// This is the line you should stop at for catch')

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24538, clang-cl does not support throw or catch")
    def test(self):
        """Test lldb exception breakpoint command for CPP."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        exception_bkpt = target.BreakpointCreateForException(
            lldb.eLanguageTypeC_plus_plus, True, True)
        self.assertTrue(exception_bkpt, "Made an exception breakpoint")

        # Now run, and make sure we hit our breakpoint:
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, "Got a valid process")

        stopped_threads = []
        stopped_threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, exception_bkpt)
        self.assertTrue(
            len(stopped_threads) == 1,
            "Stopped at our exception breakpoint.")
        thread = stopped_threads[0]
        # Make sure our throw function is still above us on the stack:

        frame_functions = lldbutil.get_function_names(thread)
        self.assertTrue(
            frame_functions.count("throws_exception_on_even(int)") == 1,
            "Our throw function is still on the stack.")

        # Okay we hit our exception throw breakpoint, now make sure we get our catch breakpoint.
        # One potential complication is that we might hit a couple of the exception breakpoints in getting out of the throw.
        # so loop till we don't see the throws function on the stack.  We should stop one more time for our exception breakpoint
        # and that should be the catch...

        while frame_functions.count("throws_exception_on_even(int)") == 1:
            stopped_threads = lldbutil.continue_to_breakpoint(
                process, exception_bkpt)
            self.assertEquals(len(stopped_threads), 1)

            thread = stopped_threads[0]
            frame_functions = lldbutil.get_function_names(thread)

        self.assertTrue(
            frame_functions.count("throws_exception_on_even(int)") == 0,
            "At catch our throw function is off the stack")
        self.assertTrue(
            frame_functions.count("intervening_function(int)") == 0,
            "At catch our intervening function is off the stack")
        self.assertTrue(
            frame_functions.count("catches_exception(int)") == 1,
            "At catch our catch function is on the stack")
