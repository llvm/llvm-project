"""Test that types defined in shared libraries work correctly."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class SharedLibTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = "shared.c"
        self.line = line_number(self.source, "// Set breakpoint 0 here.")
        self.shlib_names = ["foo"]

    def common_setup(self):
        # Run in synchronous mode
        self.dbg.SetAsync(False)
        self.runCmd("settings set symbols.load-on-demand true")

        # Create a target by the debugger.
        self.target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(self.target, VALID_TARGET)

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        self.environment = self.registerSharedLibrariesWithTarget(
            self.target, self.shlib_names
        )

        ctx = self.platformContext
        self.shared_lib_name = ctx.shlib_prefix + "foo." + ctx.shlib_extension

    @skipIfWindows
    def test_source_line_breakpoint(self):
        self.build()
        self.common_setup()

        lldbutil.run_break_set_by_file_and_line(
            self, "foo.c", 4, num_expected_locations=1, loc_exact=True
        )

        # Now launch the process, and do not stop at entry point.
        process = self.target.LaunchSimple(
            None, self.environment, self.get_process_working_directory()
        )
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )
        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

        thread = process.GetSelectedThread()
        stack_frames = thread.frames
        self.assertGreater(len(stack_frames), 2)

        leaf_frame = stack_frames[0]
        self.assertEqual("foo.c", leaf_frame.GetLineEntry().GetFileSpec().GetFilename())
        self.assertEqual(4, leaf_frame.GetLineEntry().GetLine())

        parent_frame = stack_frames[1]
        self.assertEqual(
            "shared.c", parent_frame.GetLineEntry().GetFileSpec().GetFilename()
        )
        self.assertEqual(7, parent_frame.GetLineEntry().GetLine())

    @skipIfWindows
    def test_symbolic_breakpoint(self):
        self.build()
        self.common_setup()

        lldbutil.run_break_set_by_symbol(
            self, "foo", sym_exact=True, num_expected_locations=1
        )

        # Now launch the process, and do not stop at entry point.
        process = self.target.LaunchSimple(
            None, self.environment, self.get_process_working_directory()
        )
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )
        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

        thread = process.GetSelectedThread()
        stack_frames = thread.frames
        self.assertGreater(len(stack_frames), 2)

        leaf_frame = stack_frames[0]
        self.assertEqual("foo.c", leaf_frame.GetLineEntry().GetFileSpec().GetFilename())
        self.assertEqual(4, leaf_frame.GetLineEntry().GetLine())

        parent_frame = stack_frames[1]
        self.assertEqual(
            "shared.c", parent_frame.GetLineEntry().GetFileSpec().GetFilename()
        )
        self.assertEqual(7, parent_frame.GetLineEntry().GetLine())

    @skipIfWindows
    def test_global_variable_hydration(self):
        self.build()
        self.common_setup()

        lldbutil.run_break_set_by_file_and_line(
            self, self.source, self.line, num_expected_locations=1, loc_exact=True
        )

        # Now launch the process, and do not stop at entry point.
        process = self.target.LaunchSimple(
            None, self.environment, self.get_process_working_directory()
        )
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

        self.expect(
            "target variable --shlib a.out",
            "Breakpoint in a.out should have hydrated the debug info",
            substrs=["global_shared = 897"],
        )

        self.expect(
            "target variable --shlib " + self.shared_lib_name,
            "shared library should not have debug info by default",
            matching=False,
            substrs=["global_foo"],
        )

        self.expect(
            "target variable global_foo --shlib " + self.shared_lib_name,
            "Match global_foo in symbol table should hydrate debug info",
            matching=True,
            substrs=["global_foo = 321"],
        )
