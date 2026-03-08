"""
Use lldb Python SBBlock API to access specific scopes within a frame.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FrameBlocksTestCase(TestBase):
    def test_block_equality(self):
        """Exercise SBBlock equality checks across frames and functions in different dylibs."""
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// breakpoint 1", lldb.SBFileSpec("main.c"), extra_images=["libfn"]
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame must be valid")

        main_frame_block = frame.GetFrameBlock()

        threads = lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "// breakpoint 2",
            lldb.SBFileSpec("fn.c"),
        )
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
