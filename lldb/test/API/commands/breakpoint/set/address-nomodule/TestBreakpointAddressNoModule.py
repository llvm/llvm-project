import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def get_address_from_symbol(self, symbol):
        target = lldbutil.run_to_breakpoint_make_target(self, "a.out", True)
        bp = target.BreakpointCreateByName(symbol, None)
        address = bp.GetLocationAtIndex(0).GetAddress().GetFileAddress()
        return address

    def test_set_address_no_module(self):
        self.build()

        main_address = self.get_address_from_symbol("main")

        target = lldbutil.run_to_breakpoint_make_target(self)
        debugger = target.GetDebugger()

        debugger.HandleCommand(f"break set -a {main_address:#x}")
        self.assertEquals(target.GetNumBreakpoints(), 1)

        bp = target.GetBreakpointAtIndex(0)
        self.assertIsNotNone(bp)

        _, _, thread, _ = lldbutil.run_to_breakpoint_do_run(self, target, bp)
        self.assertGreaterEqual(thread.GetNumFrames(), 1)

        thread_pc = thread.GetFrameAtIndex(0).GetPCAddress()
        self.assertNotEqual(thread_pc, None)

        self.assertEquals(main_address, thread_pc.GetFileAddress())
