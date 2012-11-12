"""
Test "print object" where another thread blocks the print object from making progress.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class PrintObjTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "print-obj")

    @dsym_test
    def test_print_obj_with_dsym(self):
        """Test "print object" where another thread blocks the print object from making progress."""
        d = {'EXE': 'a.out'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.print_obj('a.out')

    @dwarf_test
    def test_print_obj_with_dwarf(self):
        """Test "print object" where another thread blocks the print object from making progress."""
        d = {'EXE': 'b.out'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.print_obj('b.out')

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # My source program.
        self.source = "blocked.m"
        # Find the line numbers to break at.
        self.line = line_number(self.source, '// Set a breakpoint here.')

    def print_obj(self, exe_name):
        """
        Test "print object" where another thread blocks the print object from making progress.

        Set a breakpoint on the line in my_pthread_routine.  Then switch threads
        to the main thread, and do print the lock_me object.  Since that will
        try to get the lock already gotten by my_pthread_routime thread, it will
        have to switch to running all threads, and that should then succeed.
        """
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.runCmd("thread backtrace all")

        # Let's get the current stopped thread.  We'd like to switch to the
        # other thread to issue our 'po lock_me' command.
        import lldbutil
        this_thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(this_thread)

        # Find the other thread.  The iteration protocol of SBProcess and the
        # rich comparison methods (__eq__/__ne__) of SBThread come in handy.
        other_thread = None
        for t in process:
            if t != this_thread:
                other_thread = t
                break

        # Set the other thread as the selected thread to issue our 'po' command.other
        self.assertTrue(other_thread)
        process.SetSelectedThread(other_thread)
        if self.TraceOn():
            print "selected thread:" + lldbutil.get_description(other_thread)
        self.runCmd("thread backtrace")

        # We want to traverse the frame to the one corresponding to blocked.m to
        # issue our 'po lock_me' command.

        depth = other_thread.GetNumFrames()
        for i in range(depth):
            frame = other_thread.GetFrameAtIndex(i)
            name = frame.GetFunctionName()
            if name == 'main':
                other_thread.SetSelectedFrame(i)
                if self.TraceOn():
                    print "selected frame:" + lldbutil.get_description(frame)
                break

        self.expect("po lock_me", OBJECT_PRINTED_CORRECTLY,
            substrs = ['I am pretty special.'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
