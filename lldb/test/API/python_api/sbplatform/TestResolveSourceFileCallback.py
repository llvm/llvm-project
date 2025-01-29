"""
Test resolve source file callback functionality
"""

import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from pathlib import Path

import lldb

SOURCE_ORIGINAL_FILE = "test.cpp"  # File does not exist
SOURCE_NEW_FILE = "test_new.cpp"  # File exists
SOURCE_NEW_NON_EXISTENT_FILE = "non-existent-file"
EXE_NAME = "test.exe"


class ResolveSourceFileCallbackTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)

        # Set the input directory
        self.input_dir = (Path(self.getSourceDir())).resolve()

        # Set executable to test.exe and ensure it exists
        exe_path = (self.input_dir / EXE_NAME).resolve()
        self.assertTrue(exe_path.exists())
        exe_path_str = str(exe_path)

        # Create target
        self.target = self.dbg.CreateTarget(exe_path_str)
        self.assertTrue(self.target)

        # Create platform
        self.platform = self.target.GetPlatform()

        # Launch the process once, stop at breakpoint "sum" function and get the frame
        self.frame = self.get_frame_for_paused_process("sum", exe_path_str)

        # Set the original source file spec
        source_file_path = os.path.join(self.input_dir, SOURCE_ORIGINAL_FILE)
        self.original_source_file_spec = lldb.SBFileSpec(source_file_path)

        # Set the new source file spec
        new_source_file_path = os.path.join(self.input_dir, SOURCE_NEW_FILE)
        self.new_source_file_spec = lldb.SBFileSpec(new_source_file_path)

    def get_frame_for_paused_process(self, function_name, exe) -> lldb.SBFrame:
        # Launch the process, stop at breakpoint on function name and get the frame

        # Set breakpoint
        breakpoint = self.target.BreakpointCreateByName(function_name, exe)
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        # Now launch the process, and do not stop at entry point.
        process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get the stopped thread
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(), "There should be a thread stopped due to breakpoint"
        )

        # Get the frame
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid(), "There should be a valid frame")

        return frame0

    def get_source_file_for_frame(self) -> lldb.SBFileSpec:
        line_entry = self.frame.GetLineEntry()
        self.assertTrue(line_entry.IsValid(), "There should be a valid line entry")

        return line_entry.GetFileSpec()

    def test_set_non_callable(self):
        # The callback should be callable.
        non_callable = "a"

        with self.assertRaises(TypeError, msg="Need a callable object or None!"):
            self.platform.SetResolveSourceFileCallback(non_callable)

    def test_set_wrong_args(self):
        # The callback should accept 3 argument.
        def test_args2(a, b):
            pass

        with self.assertRaises(TypeError, msg="Expected 3 argument callable object"):
            self.platform.SetResolveSourceFileCallback(test_args2)

    def test_default(self):
        # The default behavior is to locate the source file with LLDB implementation
        # and frame.GetLineEntry should return the original file spec.
        resolved_source_file_spec = self.get_source_file_for_frame()

        # Check if the source file spec is resolved to the original file spec
        self.assertEqual(resolved_source_file_spec, self.original_source_file_spec)
        self.assertFalse(self.original_source_file_spec.Exists())

    def test_set_none(self):
        # SetResolveSourceFileCallback should succeed to clear the callback with None
        # and frame.GetLineEntry will return the original file spec.
        self.assertTrue(self.platform.SetResolveSourceFileCallback(None).Success())

        resolved_source_file_spec = self.get_source_file_for_frame()

        # Check if the source file spec is resolved to the original file spec
        self.assertEqual(resolved_source_file_spec, self.original_source_file_spec)
        self.assertFalse(resolved_source_file_spec.Exists())

    def test_return_original_file_on_error(self):
        # The callback fails, frame.GetLineEntry should return the original file spec.

        # Resolve Source File Callback
        def test_source_file_callback(
            module_sp: lldb.SBModule,
            original_file_spec: lldb.SBFileSpec,
            resolved_file_spec: lldb.SBFileSpec,
        ):
            return lldb.SBError("Resolve Source File Callback failed")

        self.assertTrue(
            self.platform.SetResolveSourceFileCallback(
                test_source_file_callback
            ).Success()
        )

        resolved_source_file_spec = self.get_source_file_for_frame()

        # Check if the source file spec is resolved to the original file spec
        self.assertEqual(resolved_source_file_spec, self.original_source_file_spec)
        self.assertFalse(resolved_source_file_spec.Exists())

    def test_return_orignal_file_with_new_nonexistent_file(self):
        # The callback should return a valid SBFileSpec but the file does not exist.
        # frame.GetLineEntry should return the original file spec.

        # Resolve Source File Callback
        def test_source_file_callback(
            module_sp: lldb.SBModule,
            original_file_spec: lldb.SBFileSpec,
            resolved_file_spec: lldb.SBFileSpec,
        ):
            resolved_file_spec.SetDirectory(str(self.input_dir))
            resolved_file_spec.SetFilename(SOURCE_NEW_NON_EXISTENT_FILE)

            return lldb.SBError()

        # SetResolveSourceFileCallback should succeed and frame.GetLineEntry will return the original file spec
        self.assertTrue(
            self.platform.SetResolveSourceFileCallback(
                test_source_file_callback
            ).Success()
        )

        # Get resolved source file spec from frame0
        resolved_source_file_spec = self.get_source_file_for_frame()

        # Check if the source file spec is resolved to the original file spec
        self.assertEqual(resolved_source_file_spec, self.original_source_file_spec)
        self.assertFalse(resolved_source_file_spec.Exists())

    def test_return_new_existent_file(self):
        # The callback should return a valid SBFileSpec and file exists.
        # frame.GetLineEntry should return the new file spec.

        # Resolve Source File Callback
        def test_source_file_callback(
            module_sp: lldb.SBModule,
            original_file_spec: lldb.SBFileSpec,
            resolved_file_spec: lldb.SBFileSpec,
        ):
            resolved_file_spec.SetDirectory(str(self.input_dir))
            resolved_file_spec.SetFilename(SOURCE_NEW_FILE)

            return lldb.SBError()

        # SetResolveSourceFileCallback should succeed and frame.GetLineEntry will return the new file spec from callback
        self.assertTrue(
            self.platform.SetResolveSourceFileCallback(
                test_source_file_callback
            ).Success()
        )

        # Get resolved source file spec from frame0
        resolved_source_file_spec = self.get_source_file_for_frame()

        # Check if the source file spec is resolved to the file set in callback
        self.assertEqual(resolved_source_file_spec, self.new_source_file_spec)
        self.assertFalse(self.original_source_file_spec.Exists())
        self.assertTrue(resolved_source_file_spec.Exists())
