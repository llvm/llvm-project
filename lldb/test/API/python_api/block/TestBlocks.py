"""
Use lldb Python SBBlock API to access specific scopes within a frame.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BlockAPITestCase(TestBase):
    def test_block_equality(self):
        """Exercise SBBlock equality checks."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        line1 = line_number("main.c", "// breakpoint 1")
        line2 = line_number("fn.c", "// breakpoint 2")
        breakpoint1 = target.BreakpointCreateByLocation("main.c", line1)
        breakpoint2 = target.BreakpointCreateByLocation("fn.c", line2)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint1)
        self.assertEqual(
            len(threads), 1, "There should be a thread stopped at breakpoint 1"
        )

        thread = threads[0]
        self.assertTrue(thread.IsValid(), "Thread must be valid")
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame must be valid")

        main_frame_block = frame.GetFrameBlock()

        # Continue to breakpoint 2
        process.Continue()

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint2)
        self.assertEqual(
            len(threads), 1, "There should be a thread stopped at breakpoint 2"
        )

        thread = threads[0]
        self.assertTrue(thread.IsValid(), "Thread must be valid")
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame must be valid")

        fn_frame_block = frame.GetFrameBlock()
        fn_inner_block = frame.GetBlock()

        # Check __eq__ / __ne__
        self.assertNotEqual(lldb.SBBlock(), lldb.SBBlock())
        self.assertNotEqual(fn_inner_block, fn_frame_block)
        self.assertNotEqual(main_frame_block, fn_frame_block)
        self.assertEqual(fn_inner_block.GetParent(), fn_frame_block)
        self.assertEqual(fn_frame_block.GetFirstChild(), fn_inner_block)

        # Load the main function with a different API to ensure we have two
        # distinct SBBlock objects.
        main_func_list = target.FindModule(target.GetExecutable()).FindFunctions("main")
        self.assertEqual(main_func_list.GetSize(), 1)
        main_func = main_func_list.GetContextAtIndex(0).GetFunction()
        self.assertIsNotNone(main_func)

        # Ensure they we have two distinct objects that represent the same block
        main_func_block = main_func.GetBlock()
        self.assertIsNot(main_func_block, main_frame_block)
        self.assertEqual(main_func_block, main_frame_block)
