import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    @skipIfRemote
    @skipIfWindows
    # glibc's dlopen doesn't support opening executables.
    # https://sourceware.org/bugzilla/show_bug.cgi?id=11754
    @skipIfLinux
    # freebsd's dlopen ditto
    @expectedFailureAll(oslist=["freebsd"])
    @expectedFailureNetBSD
    @no_debug_info_test
    def test(self):
        self.build()
        # Launch and stop before the dlopen call.
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        # Delete the breakpoint we no longer need.
        self.target().DeleteAllBreakpoints()

        # Check that the executable is the test binary.
        self.assertEqual(self.target().GetExecutable().GetFilename(), "a.out")

        # Continue so that dlopen is called.
        breakpoint = self.target().BreakpointCreateBySourceRegex(
            "// break after dlopen", lldb.SBFileSpec("main.c"))
        self.assertNotEqual(breakpoint.GetNumResolvedLocations(), 0)
        stopped_threads = lldbutil.continue_to_breakpoint(self.process(), breakpoint)
        self.assertEqual(len(stopped_threads), 1)

        # Check that the executable is still the test binary and not "other".
        self.assertEqual(self.target().GetExecutable().GetFilename(), "a.out")

        # Kill the process and run the program again.
        err = self.process().Kill()
        self.assertSuccess(err)

        # Test that we hit the breakpoint after dlopen.
        lldbutil.run_to_breakpoint_do_run(self, self.target(), breakpoint)
