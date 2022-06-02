"""
Test that the save_crashlog command functions
"""


import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSaveCrashlog(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_save_crashlog(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.save_crashlog()

    def save_crashlog(self):

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "I was called", self.main_source_file)

        self.runCmd("command script import lldb.macosx.crashlog")
        out_file = os.path.join(self.getBuildDir(), "crash.log")
        self.runCmd("save_crashlog '%s'"%(out_file))

        # Make sure we wrote the file:
        self.assertTrue(os.path.exists(out_file), "We wrote our file")
        
        # Now scan the file to make sure it looks right:
        # First get a few facts we'll use:
        exe_module = target.FindModule(target.GetExecutable())
        uuid_str = exe_module.GetUUIDString()

        # We'll set these to true when we find the elements in the file
        found_call_me = False
        found_main_line = False
        found_thread_header = False
        found_uuid_str = False

        with open(out_file, "r") as f:
            # We want to see a line with
            for line in f:
                if "Thread 0:" in line:
                    found_thread_header = True
                if "call_me" in line and "main.c:" in line:
                    found_call_me = True
                if "main" in line and "main.c:" in line:
                    found_main_line = True
                if uuid_str in line and "a.out" in line:
                    found_uuid_str = True
        
        self.assertTrue(found_thread_header, "Found thread header")
        self.assertTrue(found_call_me, "Found call_me line in stack")
        self.assertTrue(found_uuid_str, "Found main binary UUID")
        self.assertTrue(found_main_line, "Found main line in call stack")
                        
