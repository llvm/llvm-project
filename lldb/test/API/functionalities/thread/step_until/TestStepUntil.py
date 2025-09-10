"""Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test_event.build_exception import BuildError


class StepUntilTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"
        self.less_than_two = line_number("main.c", "Less than 2")
        self.greater_than_two = line_number("main.c", "Greater than or equal to 2.")
        self.back_out_in_main = line_number("main.c", "Back out in main")
        self.in_foo = line_number("main.c", "In foo")

    def _build_dict_for_discontinuity(self):
        return dict(
            CFLAGS_EXTRAS="-funique-basic-block-section-names "
            + "-ffunction-sections -fbasic-block-sections=list="
            + self.getSourcePath("function.list"),
            LD_EXTRAS="-Wl,--script=" + self.getSourcePath("symbol.order"),
        )

    def _common_setup(self, build_dict, args):
        self.build(dictionary=build_dict)
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        main_source_spec = lldb.SBFileSpec(self.main_source)
        break_before = target.BreakpointCreateBySourceRegex(
            "At the start", main_source_spec
        )
        self.assertTrue(break_before, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(args, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, break_before)

        if len(threads) != 1:
            self.fail("Failed to stop at first breakpoint in main.")

        thread = threads[0]
        return thread

    def do_until(self, args, until_lines, expected_linenum):
        thread = self._common_setup(None, args)

        cmd_interp = self.dbg.GetCommandInterpreter()
        ret_obj = lldb.SBCommandReturnObject()

        cmd_line = "thread until"
        for line_num in until_lines:
            cmd_line += " %d" % (line_num)

        cmd_interp.HandleCommand(cmd_line, ret_obj)
        self.assertTrue(
            ret_obj.Succeeded(), "'%s' failed: %s." % (cmd_line, ret_obj.GetError())
        )

        frame = thread.frames[0]
        line = frame.GetLineEntry().GetLine()
        self.assertEqual(
            line, expected_linenum, "Did not get the expected stop line number"
        )

    def test_hitting_one(self):
        """Test thread step until - targeting one line and hitting it."""
        self.do_until(None, [self.less_than_two], self.less_than_two)

    def test_targetting_two_hitting_first(self):
        """Test thread step until - targeting two lines and hitting one."""
        self.do_until(
            ["foo", "bar", "baz"],
            [self.less_than_two, self.greater_than_two],
            self.greater_than_two,
        )

    def test_targetting_two_hitting_second(self):
        """Test thread step until - targeting two lines and hitting the other one."""
        self.do_until(
            None, [self.less_than_two, self.greater_than_two], self.less_than_two
        )

    def test_missing_one(self):
        """Test thread step until - targeting one line and missing it by stepping out to call site"""
        self.do_until(
            ["foo", "bar", "baz"], [self.less_than_two], self.back_out_in_main
        )

    @no_debug_info_test
    def test_bad_line(self):
        """Test that we get an error if attempting to step outside the current
        function"""
        thread = self._common_setup(None, None)
        self.expect(
            f"thread until {self.in_foo}",
            substrs=["Until target outside of the current function"],
            error=True,
        )

    @no_debug_info_test
    @skipIf(oslist=lldbplatformutil.getDarwinOSTriples() + ["windows"])
    @skipIf(archs=no_match(["x86_64", "aarch64"]))
    @skipIf(compiler=no_match(["clang"]))
    def test_bad_line_discontinuous(self):
        """Test that we get an error if attempting to step outside the current
        function -- and the function is discontinuous"""
        try:
            self.build(dictionary=self._build_dict_for_discontinuity())
        except BuildError as ex:
            self.skipTest(f"failed to build with linker script.")

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "At the start", lldb.SBFileSpec(self.main_source)
        )
        self.expect(
            f"thread until {self.in_foo}",
            substrs=["Until target outside of the current function"],
            error=True,
        )
