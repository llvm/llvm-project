"""
Test SBFrameList API.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FrameListAPITestCase(TestBase):
    def test_frame_list_api(self):
        """Test SBThread.GetFrames() returns a valid SBFrameList."""
        self.build()
        self.frame_list_api()

    def test_frame_list_iterator(self):
        """Test SBFrameList iterator functionality."""
        self.build()
        self.frame_list_iterator()

    def test_frame_list_indexing(self):
        """Test SBFrameList indexing and length."""
        self.build()
        self.frame_list_indexing()

    def test_frame_list_get_thread(self):
        """Test SBFrameList.GetThread() returns correct thread."""
        self.build()
        self.frame_list_get_thread()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.cpp"

    def frame_list_api(self):
        """Test SBThread.GetFrames() returns a valid SBFrameList."""
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line", lldb.SBFileSpec(self.main_source)
        )

        self.assertTrue(
            thread.IsValid(), "There should be a thread stopped due to breakpoint"
        )

        # Test GetFrames() returns a valid SBFrameList
        frame_list = thread.GetFrames()
        self.assertTrue(frame_list.IsValid(), "Frame list should be valid")
        self.assertGreater(
            frame_list.GetSize(), 0, "Frame list should have at least one frame"
        )

        # Verify frame list size matches thread frame count
        self.assertEqual(
            frame_list.GetSize(),
            thread.GetNumFrames(),
            "Frame list size should match thread frame count",
        )

        # Verify frames are the same
        for i in range(frame_list.GetSize()):
            frame_from_list = frame_list.GetFrameAtIndex(i)
            frame_from_thread = thread.GetFrameAtIndex(i)
            self.assertTrue(
                frame_from_list.IsValid(), f"Frame {i} from list should be valid"
            )
            self.assertEqual(
                frame_from_list.GetPC(),
                frame_from_thread.GetPC(),
                f"Frame {i} PC should match",
            )

    def frame_list_iterator(self):
        """Test SBFrameList iterator functionality."""
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line", lldb.SBFileSpec(self.main_source)
        )

        self.assertTrue(
            thread.IsValid(), "There should be a thread stopped due to breakpoint"
        )

        frame_list = thread.GetFrames()

        # Test iteration
        frame_count = 0
        for frame in frame_list:
            self.assertTrue(frame.IsValid(), "Each frame should be valid")
            frame_count += 1

        self.assertEqual(
            frame_count,
            frame_list.GetSize(),
            "Iterator should visit all frames",
        )

        # Test that we can iterate multiple times
        second_count = 0
        for frame in frame_list:
            second_count += 1

        self.assertEqual(
            frame_count, second_count, "Should be able to iterate multiple times"
        )

    def frame_list_indexing(self):
        """Test SBFrameList indexing and length."""
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line", lldb.SBFileSpec(self.main_source)
        )

        self.assertTrue(
            thread.IsValid(), "There should be a thread stopped due to breakpoint"
        )

        frame_list = thread.GetFrames()

        # Test len()
        self.assertEqual(
            len(frame_list), frame_list.GetSize(), "len() should return frame count"
        )

        # Test positive indexing
        first_frame = frame_list[0]
        self.assertTrue(first_frame.IsValid(), "First frame should be valid")
        self.assertEqual(
            first_frame.GetPC(),
            thread.GetFrameAtIndex(0).GetPC(),
            "Indexed frame should match",
        )

        # Test negative indexing
        if len(frame_list) > 0:
            last_frame = frame_list[-1]
            self.assertTrue(last_frame.IsValid(), "Last frame should be valid")
            self.assertEqual(
                last_frame.GetPC(),
                thread.GetFrameAtIndex(len(frame_list) - 1).GetPC(),
                "Negative indexing should work",
            )

        # Test out of bounds returns None
        out_of_bounds = frame_list[10000]
        self.assertIsNone(out_of_bounds, "Out of bounds index should return None")

        # Test bool conversion
        self.assertTrue(bool(frame_list), "Non-empty frame list should be truthy")

        # Test Clear()
        frame_list.Clear()
        # Note: Clear() clears the underlying StackFrameList cache,
        # but the frame list object itself should still be valid
        self.assertTrue(
            frame_list.IsValid(), "Frame list should still be valid after Clear()"
        )

    def frame_list_get_thread(self):
        """Test SBFrameList.GetThread() returns correct thread."""
        exe = self.getBuildArtifact("a.out")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line", lldb.SBFileSpec(self.main_source)
        )

        self.assertTrue(
            thread.IsValid(), "There should be a thread stopped due to breakpoint"
        )

        frame_list = thread.GetFrames()
        self.assertTrue(frame_list.IsValid(), "Frame list should be valid")

        # Test GetThread() returns the correct thread
        thread_from_list = frame_list.GetThread()
        self.assertTrue(
            thread_from_list.IsValid(), "Thread from frame list should be valid"
        )
        self.assertEqual(
            thread_from_list.GetThreadID(),
            thread.GetThreadID(),
            "Frame list should return the correct thread",
        )

        # Verify it's the same thread object
        self.assertEqual(
            thread_from_list.GetProcess().GetProcessID(),
            thread.GetProcess().GetProcessID(),
            "Thread should belong to same process",
        )
