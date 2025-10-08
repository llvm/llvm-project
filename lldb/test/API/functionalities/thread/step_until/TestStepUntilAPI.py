from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test_event.build_exception import BuildError


class TestStepUntilAPI(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super().setUp()

        self.main_source = "main.c"
        self.main_spec = lldb.SBFileSpec(self.main_source)
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

    def _do_until(self, build_dict, args, until_line, expected_line):
        self.build(dictionary=build_dict)
        launch_info = lldb.SBLaunchInfo(args)
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "At the start", self.main_spec, launch_info
        )

        self.assertSuccess(
            thread.StepOverUntil(self.frame(), self.main_spec, until_line)
        )

        self.runCmd("process status")

        line = self.frame().GetLineEntry().GetLine()
        self.assertEqual(
            line, expected_line, "Did not get the expected stop line number"
        )

    def _assertDiscontinuity(self):
        target = self.target()
        foo = target.FindFunctions("foo")
        self.assertEqual(len(foo), 1)
        foo = foo[0]

        call_me = self.target().FindFunctions("call_me")
        self.assertEqual(len(call_me), 1)
        call_me = call_me[0]

        foo_addr = foo.function.GetStartAddress().GetLoadAddress(target)
        found_before = False
        found_after = False
        for range in call_me.function.GetRanges():
            addr = range.GetBaseAddress().GetLoadAddress(target)
            if addr < foo_addr:
                found_before = True
            if addr > foo_addr:
                found_after = True

        self.assertTrue(
            found_before and found_after,
            "'foo' is not between 'call_me'" + str(foo) + str(call_me),
        )

    def test_hitting(self):
        """Test SBThread.StepOverUntil - targeting a line and hitting it."""
        self._do_until(None, None, self.less_than_two, self.less_than_two)

    @skipIf(oslist=lldbplatformutil.getDarwinOSTriples() + ["windows"])
    @skipIf(archs=no_match(["x86_64", "aarch64"]))
    @skipIf(compiler=no_match(["clang"]))
    def test_hitting_discontinuous(self):
        """Test SBThread.StepOverUntil - targeting a line and hitting it -- with
        discontinuous functions"""
        try:
            self._do_until(
                self._build_dict_for_discontinuity(),
                None,
                self.less_than_two,
                self.less_than_two,
            )
        except BuildError as ex:
            self.skipTest(f"failed to build with linker script.")

        self._assertDiscontinuity()

    def test_missing(self):
        """Test SBThread.StepOverUntil - targeting a line and missing it by stepping out to call site"""
        self._do_until(
            None, ["foo", "bar", "baz"], self.less_than_two, self.back_out_in_main
        )

    @skipIf(oslist=lldbplatformutil.getDarwinOSTriples() + ["windows"])
    @skipIf(archs=no_match(["x86_64", "aarch64"]))
    @skipIf(compiler=no_match(["clang"]))
    def test_missing_discontinuous(self):
        """Test SBThread.StepOverUntil - targeting a line and missing it by
        stepping out to call site -- with discontinuous functions"""
        try:
            self._do_until(
                self._build_dict_for_discontinuity(),
                ["foo", "bar", "baz"],
                self.less_than_two,
                self.back_out_in_main,
            )
        except BuildError as ex:
            self.skipTest(f"failed to build with linker script.")

        self._assertDiscontinuity()

    def test_bad_line(self):
        """Test that we get an error if attempting to step outside the current
        function"""
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "At the start", self.main_spec
        )
        self.assertIn(
            "step until target not in current function",
            thread.StepOverUntil(
                self.frame(), self.main_spec, self.in_foo
            ).GetCString(),
        )

    @skipIf(oslist=lldbplatformutil.getDarwinOSTriples() + ["windows"])
    @skipIf(archs=no_match(["x86_64", "aarch64"]))
    @skipIf(compiler=no_match(["clang"]))
    def test_bad_line_discontinuous(self):
        """Test that we get an error if attempting to step outside the current
        function -- and the function is discontinuous"""

        try:
            self.build(dictionary=self._build_dict_for_discontinuity())
            _, _, thread, _ = lldbutil.run_to_source_breakpoint(
                self, "At the start", self.main_spec
            )
        except BuildError as ex:
            self.skipTest(f"failed to build with linker script.")

        self.assertIn(
            "step until target not in current function",
            thread.StepOverUntil(
                self.frame(), self.main_spec, self.in_foo
            ).GetCString(),
        )
        self._assertDiscontinuity()
