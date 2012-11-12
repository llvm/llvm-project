"""
Test lldb exception breakpoint command for CPP.
"""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class CPPBreakpointTestCase(TestBase):

    mydir = os.path.join("lang", "cpp", "exceptions")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test lldb exception breakpoint command for CPP."""
        self.buildDsym()
        self.cpp_exceptions()

    @dwarf_test
    def test_with_dwarf(self):
        """Test lldb exception breakpoint command for CPP."""
        self.buildDwarf()
        self.cpp_exceptions()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.source = 'exceptions.cpp'
        self.catch_line = line_number(self.source, '// This is the line you should stop at for catch')

    def cpp_exceptions (self):
        """Test lldb exception breakpoint command for CPP."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget (exe)
        self.assertTrue(target, VALID_TARGET)

        exception_bkpt = target.BreakpointCreateForException (lldb.eLanguageTypeC_plus_plus, True, True)
        self.assertTrue (exception_bkpt, "Made an exception breakpoint")

        # Now run, and make sure we hit our breakpoint:
        process = target.LaunchSimple (None, None, os.getcwd())
        self.assertTrue (process, "Got a valid process")
        
        stopped_threads = []
        stopped_threads = lldbutil.get_threads_stopped_at_breakpoint (process, exception_bkpt)
        self.assertTrue (len(stopped_threads) == 1, "Stopped at our exception breakpoint.")
        thread = stopped_threads[0]
        # Make sure our throw function is still above us on the stack:

        frame_functions = lldbutil.get_function_names(thread)
        self.assertTrue (frame_functions.count ("throws_exception_on_even(int)") == 1, "Our throw function is still on the stack.")

        # Okay we hit our exception throw breakpoint, now make sure we get our catch breakpoint.
        # One potential complication is that we might hit a couple of the exception breakpoints in getting out of the throw.
        # so loop till we don't see the throws function on the stack.  We should stop one more time for our exception breakpoint
        # and that should be the catch...

        while frame_functions.count ("throws_exception_on_even(int)") == 1: 
            stopped_threads = lldbutil.continue_to_breakpoint (process, exception_bkpt)
            self.assertTrue (len(stopped_threads) == 1)
        
            thread = stopped_threads[0]
            frame_functions = lldbutil.get_function_names(thread)

        self.assertTrue (frame_functions.count ("throws_exception_on_even(int)") == 0, "At catch our throw function is off the stack")
        self.assertTrue (frame_functions.count ("intervening_function(int)") == 0,     "At catch our intervening function is off the stack")
        self.assertTrue (frame_functions.count ("catches_exception(int)") == 1, "At catch our catch function is on the stack")

        
                
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
