import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

def get_num_registers(frame):
    general_purpose_regs = frame.GetRegisters()[0]
    num_regs = sum(reg.GetError().Success() for reg in general_purpose_regs)
    return num_regs

class TestSwiftAsyncUnwind(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test async unwind"""
        self.build()
        src = lldb.SBFileSpec('main.swift')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', src)

        self.assertTrue("sayolleH" in thread.GetFrameAtIndex(0).GetFunctionName(), 
                "Redundantly confirm that we're stopped in sayolleH()")

        if self.TraceOn():
           self.runCmd("bt all")

        self.assertTrue("sayHello" in thread.GetFrameAtIndex(1).GetFunctionName())
        self.assertTrue("sayGeneric" in thread.GetFrameAtIndex(2).GetFunctionName())

        # Check that we can only get a limited number of registers for
        # frames that unwound with an AsyncContext, as a sanity check
        # to see that this is really the async unwinder.
        self.assertIn(get_num_registers(thread.GetFrameAtIndex(1)), [2,3,4])
        self.assertIn(get_num_registers(thread.GetFrameAtIndex(2)), [2,3,4])

        # Delete the old breakpoint, otherwise it would be reached again.
        target.BreakpointDelete(bkpt.GetID())
        lldbutil.continue_to_source_breakpoint(
            self, process, "break synchronous hello", src
        )

        self.assertTrue(
            "synchronousSayHelo" in thread.GetFrameAtIndex(0).GetFunctionName(),
        )
        frame1 = thread.GetFrameAtIndex(1)
        self.assertTrue(
            "callSyncHello" in frame1.GetFunctionName(),
        )
        location = frame1.GetLineEntry()
        # Check that the callsite location is on the correct line.
        self.assertEqual(
            location.GetLine(), lldbtest.line_number("main.swift", "frame 1 line")
        )
