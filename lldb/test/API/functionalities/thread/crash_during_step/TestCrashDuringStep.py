"""
Test that step-inst over a crash behaves correctly.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CrashDuringStepTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.breakpoint = line_number("main.cpp", "// Set breakpoint here")

    # IO error due to breakpoint at invalid address
    @expectedFailureAll(triple=re.compile("^mips"))
    @skipIf(oslist=["windows"], archs=["aarch64"])
    def test_step_inst_with(self):
        """Test thread creation during step-inst handling."""
        self.build()
        _, process, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.cpp"), self.breakpoint
        )

        # Keep stepping until the inferior crashes
        while (
            process.GetState() == lldb.eStateStopped
            and not lldbutil.is_thread_crashed(self, thread)
        ):
            thread.StepInstruction(False)

        self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        self.assertTrue(lldbutil.is_thread_crashed(self, thread), "Thread has crashed")
        process.Kill()
