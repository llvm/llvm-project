"""
Make sure that we can step in across an arm64 branch island
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestBranchIslandStepping(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessAppleSilicon
    def test_step_in_branch_island(self):
        """Make sure we can step in across a branch island"""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.do_test()

    def do_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        # Make sure that we did manage to generate a branch island for foo:
        syms = target.FindSymbols("foo.island", lldb.eSymbolTypeCode)
        self.assertEqual(len(syms), 1, "We did generate an island for foo")

        # Gathering some info to dump in case of failure:
        trace_before = lldbutil.print_stacktrace(thread, True)
        func_before = thread.frames[0].function

        thread.StepInto()
        stop_frame = thread.frames[0]
        # This is failing on the bot, but I can't reproduce the failure
        # locally.  Let's see if we can dump some more info here to help
        # figure out what went wrong...
        if stop_frame.name.find("foo") == -1:
            stream = lldb.SBStream()
            print("Branch island symbols: ")
            syms[0].GetDescription(stream)
            for i in range(0, 6):
                for sep in ["", "."]:
                    syms = target.FindSymbols(
                        f"foo.island{sep}{i}", lldb.eSymbolTypeCode
                    )
                    if len(syms) > 0:
                        stream.Print("\n")
                        syms[0].GetDescription(stream)

            print(stream.GetData())
            print(f"Start backtrace:")
            print(trace_before)
            print(f"\n'main' disassembly:\n{lldbutil.disassemble(target, func_before)}")
            print("\nEnd backtrace:\n")
            lldbutil.print_stacktrace(thread)
            print(
                f"\nStop disassembly:\n {lldbutil.disassemble(target, stop_frame.function)}"
            )

        self.assertIn("foo", stop_frame.name, "Stepped into foo")
        var = stop_frame.FindVariable("a_variable_in_foo")
        self.assertTrue(var.IsValid(), "Found the variable in foo")
