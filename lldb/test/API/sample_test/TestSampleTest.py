"""
Describe the purpose of the test class here.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class RenameThisSampleTestTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_sample_rename_this(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.sample_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Set up your test case here. If your test doesn't need any set up then
        # remove this method from your TestCase class.

    def sample_test(self):
        """You might use the test implementation in several ways, say so here."""

        # This function starts a process, "a.out" by default, sets a source
        # breakpoint, runs to it, and returns the thread, process & target.
        # It optionally takes an SBLaunchOption argument if you want to pass
        # arguments or environment variables.
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        frame = thread.GetFrameAtIndex(0)
        test_var = frame.FindVariable("test_var")
        self.assertTrue(test_var.GetError().Success(), "Failed to fetch test_var")
        test_value = test_var.GetValueAsUnsigned()
        self.assertEqual(test_value, 10, "Got the right value for test_var")

