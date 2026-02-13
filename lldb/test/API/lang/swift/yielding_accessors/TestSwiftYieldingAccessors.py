import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftStepping(lldbtest.TestBase):
    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        lldbutil.ignore_swift_stdlib_when_stepping(platform, self)

    def check_stop_reason_plan_complete(self, thread):
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

    def check_self_available(self, thread):
        frame0 = thread.frames[0]
        self_var = frame0.FindVariable("self")
        self.assertSuccess(self_var.GetError(), "Failed to fetch self")
        member_int = self_var.GetChildAtIndex(0)
        self.assertSuccess(member_int.GetError(), "Failed to fetch self.member_int")
        self.assertEqual(member_int.GetValueAsSigned(), 42)

    def hit_correct_line(self, thread, pattern):
        self.check_stop_reason_plan_complete(thread)
        target_line = lldbtest.line_number(self.main_source, pattern)
        self.assertNotEqual(target_line, 0, "Could not find source pattern " + pattern)
        cur_line = thread.frames[0].GetLineEntry().GetLine()
        hit_line = cur_line == target_line
        self.assertTrue(
            hit_line,
            "Stepped to line %d instead of expected %d "
            "with pattern '%s'\nBacktrace = \n%s."
            % (
                cur_line,
                target_line,
                pattern,
                "\n".join(str(frame) for frame in thread.frames),
            ),
        )
        return hit_line

    @swiftTest
    def test_correct_number_of_breakpoints(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        breakpoint = target.BreakpointCreateBySourceRegex(
            "yield line", self.main_source_spec
        )
        self.assertEqual(breakpoint.GetNumLocations(), 1, breakpoint)

    @swiftTest
    def test_step_over_starting_inside_coroutine(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "first coroutine line", self.main_source_spec
        )
        self.check_self_available(thread)
        thread.StepOver()
        self.hit_correct_line(thread, "yield line")
        self.check_self_available(thread)
        thread.StepOver()
        self.hit_correct_line(thread, "coroutine call site")
        thread.StepOver()
        self.hit_correct_line(thread, "last main line")

    @swiftTest
    def test_step_in_and_out_callsite(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "coroutine call site", self.main_source_spec
        )
        thread.StepInto()
        self.hit_correct_line(thread, "first coroutine line")
        self.check_self_available(thread)
        thread.StepOut()
        self.hit_correct_line(thread, "coroutine call site")
        thread.StepInto()
        self.hit_correct_line(thread, "USE line")
        thread.StepOut()
        self.hit_correct_line(thread, "coroutine call site")
        thread.StepInto()
        self.hit_correct_line(thread, "last coroutine line")
        self.check_self_available(thread)
        thread.StepOut()
        # FIXME: this last line behaves differently in CI, where it instead
        # returns to the last line of main. This is likely a result of the
        # debug symbols in the standard library.
        # self.hit_correct_line(thread, "coroutine call site")
