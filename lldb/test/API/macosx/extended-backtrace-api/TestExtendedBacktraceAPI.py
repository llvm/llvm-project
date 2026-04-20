"""Test SBThread.GetExtendedBacktraceThread API with queue debugging."""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestExtendedBacktraceAPI(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.m"

    @skipUnlessDarwin
    @add_test_categories(["objc", "pyapi"])
    def test_extended_backtrace_thread_api(self):
        """Test GetExtendedBacktraceThread with queue debugging."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Get Xcode developer directory path.
        # Try DEVELOPER_DIR environment variable first, then fall back to xcode-select.
        xcode_dev_path = os.environ.get("DEVELOPER_DIR")

        if not xcode_dev_path:
            import subprocess

            xcode_dev_path = (
                subprocess.check_output(["xcode-select", "-p"]).decode("utf-8").strip()
            )

        # Check for libBacktraceRecording.dylib.
        libbtr_path = os.path.join(
            xcode_dev_path, "usr/lib/libBacktraceRecording.dylib"
        )

        self.assertTrue(
            os.path.isfile(libbtr_path),
            f"libBacktraceRecording.dylib is not present at {libbtr_path}",
        )

        self.assertTrue(
            os.path.isfile("/usr/lib/system/introspection/libdispatch.dylib"),
            "introspection libdispatch dylib not installed.",
        )

        # Create launch info with environment variables for libBacktraceRecording.
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        launch_info.SetEnvironmentEntries(
            [
                f"DYLD_INSERT_LIBRARIES={libbtr_path}",
                "DYLD_LIBRARY_PATH=/usr/lib/system/introspection",
            ],
            True,
        )

        # Launch the process and run to breakpoint.
        target, process, thread, bp = lldbutil.run_to_name_breakpoint(
            self, "do_work_level_5", launch_info=launch_info, bkpt_module="a.out"
        )

        self.assertTrue(target.IsValid(), VALID_TARGET)
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)
        self.assertTrue(thread.IsValid(), "Stopped thread is valid")
        self.assertTrue(bp.IsValid(), VALID_BREAKPOINT)

        # Call GetNumQueues to ensure queue information is loaded.
        num_queues = process.GetNumQueues()

        # Check that we can find the com.apple.main-thread queue.
        main_thread_queue_found = False
        for i in range(num_queues):
            queue = process.GetQueueAtIndex(i)
            if queue.GetName() == "com.apple.main-thread":
                main_thread_queue_found = True
                break

        # Verify we have at least 5 frames.
        self.assertGreaterEqual(
            thread.GetNumFrames(),
            5,
            "Thread should have at least 5 frames in backtrace",
        )

        # Get frame 2 BEFORE calling GetExtendedBacktraceThread.
        # This mimics what Xcode does - it has the frame objects ready.
        frame2 = thread.GetFrameAtIndex(2)
        self.assertTrue(frame2.IsValid(), "Frame 2 is valid")

        # Now test GetExtendedBacktraceThread.
        # This is the critical part - getting the extended backtrace calls into
        # libBacktraceRecording which does an inferior function call, and this
        # invalidates/clears the unwinder state.
        extended_thread = thread.GetExtendedBacktraceThread("libdispatch")

        # This should be valid since we injected libBacktraceRecording.
        self.assertTrue(
            extended_thread.IsValid(),
            "Extended backtrace thread for 'libdispatch' should be valid with libBacktraceRecording loaded",
        )

        # The extended thread should have frames.
        self.assertGreater(
            extended_thread.GetNumFrames(),
            0,
            "Extended backtrace thread should have at least one frame",
        )

        # Test frame 2 on the extended backtrace thread.
        self.assertGreater(
            extended_thread.GetNumFrames(),
            2,
            "Extended backtrace thread should have at least 3 frames to access frame 2",
        )

        extended_frame2 = extended_thread.GetFrameAtIndex(2)
        self.assertTrue(extended_frame2.IsValid(), "Extended thread frame 2 is valid")

        # NOW try to access variables from frame 2 of the ORIGINAL thread.
        # This is the key test - after GetExtendedBacktraceThread() has executed
        # an inferior function call, the unwinder state may be invalidated.
        # Xcode exhibits this bug where variables show "register fp is not available"
        # after extended backtrace retrieval.

        # Set frame 2 as the selected frame so expect_var_path works.
        thread.SetSelectedFrame(2)

        variables = frame2.GetVariables(False, True, False, True)
        self.assertGreater(
            variables.GetSize(), 0, "Frame 2 should have at least one variable"
        )

        # Test all variables in frame 2, like Xcode does.
        # Use expect_var_path to verify each variable is accessible without errors.
        for i in range(variables.GetSize()):
            var = variables.GetValueAtIndex(i)
            var_name = var.GetName()

            # This will fail if the variable contains "not available" or has errors.
            self.expect_var_path(var_name)
