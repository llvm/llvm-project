import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibCxxStdFunctionRecognizerTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(["libc++"])
    def test_frame_recognizer(self):
        """Test that implementation details of `std::invoke` are hidden"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        stop_cnt = 0
        while process.GetState() != lldb.eStateExited:
            stop_cnt += 1
            self.assertTrue(
                any(
                    f in thread.GetFrameAtIndex(0).GetFunctionName()
                    for f in ["consume_number", "add", "Callable"]
                )
            )
            # Skip all hidden frames
            frame_id = 1
            while (
                frame_id < thread.GetNumFrames()
                and thread.GetFrameAtIndex(frame_id).IsHidden()
            ):
                frame_id = frame_id + 1
            # Expect `std::invoke` to be the direct parent
            self.assertIn(
                "::invoke", thread.GetFrameAtIndex(frame_id).GetFunctionName()
            )
            # And right above that, there should be the `main` frame
            self.assertIn(
                "main", thread.GetFrameAtIndex(frame_id + 1).GetFunctionName()
            )
            process.Continue()

        self.assertEqual(stop_cnt, 4)
