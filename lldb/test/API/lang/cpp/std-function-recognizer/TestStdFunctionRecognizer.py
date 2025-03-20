import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibCxxStdFunctionRecognizerTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(["libc++"])
    def test_frame_recognizer(self):
        """Test that std::function all implementation details are hidden in SBFrame"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        self.assertIn("foo", thread.GetFrameAtIndex(0).GetFunctionName())
        # Skip all hidden frames
        frame_id = 1
        while (
            frame_id < thread.GetNumFrames()
            and thread.GetFrameAtIndex(frame_id).IsHidden()
        ):
            frame_id = frame_id + 1
        # Expect `std::function<...>::operator()` to be the direct parent of `foo`
        self.assertIn(
            "::operator()", thread.GetFrameAtIndex(frame_id).GetFunctionName()
        )
        # And right above that, there should be the `main` frame
        self.assertIn("main", thread.GetFrameAtIndex(frame_id + 1).GetFunctionName())

    @add_test_categories(["libc++"])
    def test_backtrace(self):
        """Test that std::function implementation details are hidden in bt"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        # Filtered.
        self.expect(
            "thread backtrace",
            ordered=True,
            substrs=["frame", "foo", "frame", "main"],
        )
        self.expect(
            "thread backtrace", matching=False, patterns=["frame.*std::__.*::__function"]
        )
        # Unfiltered.
        self.expect(
            "bt -u",
            ordered=True,
            patterns=["frame.*foo", "frame.*std::__[^:]*::__function", "frame.*main"],
        )
        self.expect(
            "thread backtrace -u",
            ordered=True,
            patterns=["frame.*foo", "frame.*std::__[^:]*::__function", "frame.*main"],
        )
        self.expect(
            "thread backtrace --unfiltered",
            ordered=True,
            patterns=["frame.*foo", "frame.*std::__[^:]*::__function", "frame.*main"],
        )

    @add_test_categories(["libc++"])
    def test_up_down(self):
        """Test that std::function implementation details are skipped"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        frame = thread.selected_frame
        # up
        self.assertIn("foo", frame.GetFunctionName())
        start_idx = frame.GetFrameID()
        i = 0
        while i < thread.GetNumFrames():
            self.expect("up")
            frame = thread.selected_frame
            if frame.GetFunctionName() == "main":
                break
        end_idx = frame.GetFrameID()
        self.assertLess(i, end_idx - start_idx, "skipped frames")

        # Back down again.
        start_idx = frame.GetFrameID()
        for i in range(1, thread.GetNumFrames()):
            self.expect("down")
            frame = thread.selected_frame
            if "foo" in frame.GetFunctionName():
                break
        end_idx = frame.GetFrameID()
        self.assertLess(i, start_idx - end_idx, "skipped frames")

    @add_test_categories(["libc++"])
    def test_api(self):
        """Test that std::function implementation details are skipped"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        num_hidden = 0
        for frame in thread.frames:
            if frame.IsHidden():
                num_hidden += 1

        self.assertGreater(num_hidden, 0)
        self.assertLess(num_hidden, thread.GetNumFrames())
