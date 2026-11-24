"""
Test "print object" where another thread blocks the print object from making progress.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PrintObjTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # My source program.
        self.source = "blocked.m"
        # Find the line numbers to break at.
        self.line = line_number(self.source, "// Set a breakpoint here.")

    def test_print_obj(self):
        """
        Test "print object" where another thread blocks the print object from making progress.

        Set a breakpoint on the line in my_pthread_routine.  Then switch threads
        to the main thread, and do print the lock_me object.  Since that will
        try to get the lock already gotten by my_pthread_routime thread, it will
        have to switch to running all threads, and that should then succeed.
        """
        d = {"EXE": "b.out"}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)

        # Let's get the current stopped thread.  We'd like to switch to the
        # other thread to issue our 'po lock_me' command.
        _, process, this_thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec(self.source, False), self.line, exe_name="b.out"
        )
        self.runCmd("breakpoint list")

        self.runCmd("thread backtrace all")

        # Find the other thread.  The iteration protocol of SBProcess and the
        # rich comparison methods (__eq__/__ne__) of SBThread come in handy.
        other_thread = None
        for t in process:
            if t != this_thread:
                other_thread = t
                break

        # Set the other thread as the selected thread to issue our 'po'
        # command.other
        self.assertTrue(other_thread)
        process.SetSelectedThread(other_thread)
        if self.TraceOn():
            print("selected thread:" + lldbutil.get_description(other_thread))
        self.runCmd("thread backtrace")

        # We want to traverse the frame to the one corresponding to blocked.m to
        # issue our 'po lock_me' command.

        for frame in other_thread.frames:
            if frame.name == "main":
                other_thread.selected_frame = frame
                if self.TraceOn():
                    print("selected frame:" + lldbutil.get_description(frame))
                break

        self.expect(
            "po lock_me", OBJECT_PRINTED_CORRECTLY, substrs=["I am pretty special."]
        )
