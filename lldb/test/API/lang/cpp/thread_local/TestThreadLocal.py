from lldbsuite.test import decorators

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest


class PlatformProcessCrashInfoTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=["windows", "linux", "freebsd", "netbsd"])
    def test_thread_local(self):
        # Set a breakpoint on the first instruction of the main function,
        # before the TLS initialization has run.
        self.build()
        exe = self.getBuildArtifact("a.out")

        (target, process, _, _) = \
            lldbutil.run_to_source_breakpoint(self, "Set breakpoint here",
                                              lldb.SBFileSpec("main.cpp"))
        self.expect_expr("tl_local_int + 1",
                         result_type="int", result_value="323")
        self.expect_expr("*tl_local_ptr + 2",
                         result_type="int", result_value="324")
        self.expect_expr("tl_global_int",
                         result_type="int", result_value="123")
        self.expect_expr("*tl_global_ptr",
                         result_type="int", result_value="45")

        # Create the filespec by which to locate our a.out module.
        #
        #  - Use the absolute path to get the module for the current variant.
        #  - Use the relative path for reproducers. The modules are never
        #    orphaned because the SB objects are leaked intentionally. This
        #    causes LLDB to reuse the same module for every variant, because the
        #    UUID is the same for all the inferiors. FindModule below only
        #    compares paths and is oblivious to the fact that the UUIDs are the
        #    same.
        if configuration.is_reproducer():
            filespec = lldb.SBFileSpec('a.out', False)
        else:
            filespec = lldb.SBFileSpec(exe, False)

        # Now see if we emit the correct error when the TLS is not yet
        # initialized. Let's set a breakpoint on the first instruction
        # of main.
        main_module = target.FindModule(filespec)
        self.assertTrue(main_module, VALID_MODULE)
        main_address = main_module.FindSymbol("main").GetStartAddress()
        main_bkpt = target.BreakpointCreateBySBAddress(main_address)

        process.Kill()
        lldbutil.run_to_breakpoint_do_run(self, target, main_bkpt)

        self.expect("expr tl_local_int", error=True,
                    substrs=["couldn't get the value of variable tl_local_int",
                             "No TLS data currently exists for this thread"])
        self.expect("expr *tl_local_ptr", error=True,
                    substrs=["couldn't get the value of variable tl_local_ptr",
                             "No TLS data currently exists for this thread"])

