"""Test stepping through ObjC method dispatch in various forms."""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCStepping(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "stepping-tests.m"
        self.source_randomMethod_line = line_number(
            self.main_source, '// Source randomMethod start line.')
        self.sourceBase_randomMethod_line = line_number(
            self.main_source, '// SourceBase randomMethod start line.')
        self.source_returnsStruct_start_line = line_number(
            self.main_source, '// Source returnsStruct start line.')
        self.sourceBase_returnsStruct_start_line = line_number(
            self.main_source, '// SourceBase returnsStruct start line.')
        self.stepped_past_nil_line = line_number(
            self.main_source, '// Step over nil should stop here.')

    @add_test_categories(['pyapi', 'basic_process'])
    def test_with_python_api(self):
        """Test stepping through ObjC method dispatch in various forms."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        breakpoints_to_disable = []

        break1 = target.BreakpointCreateBySourceRegex(
            "// Set first breakpoint here.", self.main_source_spec)
        self.assertTrue(break1, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break1)

        break2 = target.BreakpointCreateBySourceRegex(
            "// Set second breakpoint here.", self.main_source_spec)
        self.assertTrue(break2, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break2)

        break3 = target.BreakpointCreateBySourceRegex(
            '// Set third breakpoint here.', self.main_source_spec)
        self.assertTrue(break3, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break3)

        break4 = target.BreakpointCreateBySourceRegex(
            '// Set fourth breakpoint here.', self.main_source_spec)
        self.assertTrue(break4, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break4)

        break5 = target.BreakpointCreateBySourceRegex(
            '// Set fifth breakpoint here.', self.main_source_spec)
        self.assertTrue(break5, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break5)

        break_returnStruct_call_super = target.BreakpointCreateBySourceRegex(
            '// Source returnsStruct call line.', self.main_source_spec)
        self.assertTrue(break_returnStruct_call_super, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break_returnStruct_call_super)

        break_step_nil = target.BreakpointCreateBySourceRegex(
            '// Set nil step breakpoint here.', self.main_source_spec)
        self.assertTrue(break_step_nil, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, break1)
        if len(threads) != 1:
            self.fail("Failed to stop at breakpoint 1.")

        thread = threads[0]

        mySource = thread.GetFrameAtIndex(0).FindVariable("mySource")
        self.assertTrue(mySource, "Found mySource local variable.")
        mySource_isa = mySource.GetChildMemberWithName("isa")
        self.assertTrue(mySource_isa, "Found mySource->isa local variable.")
        className = mySource_isa.GetSummary()

        if self.TraceOn():
            print(mySource_isa)

        # Lets delete mySource so we can check that after stepping a child variable
        # with no parent persists and is useful.
        del (mySource)

        # Now step in, that should leave us in the Source randomMethod:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(
            line_number, self.source_randomMethod_line,
            "Stepped into Source randomMethod.")

        # Now step in again, through the super call, and that should leave us
        # in the SourceBase randomMethod:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(
            line_number, self.sourceBase_randomMethod_line,
            "Stepped through super into SourceBase randomMethod.")

        threads = lldbutil.continue_to_breakpoint(process, break2)
        self.assertEqual(
            len(threads), 1,
            "Continued to second breakpoint in main.")

        # Again, step in twice gets us to a stret method and a stret super
        # call:
        thread = threads[0]
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(
            line_number, self.source_returnsStruct_start_line,
            "Stepped into Source returnsStruct.")

        threads = lldbutil.continue_to_breakpoint(
            process, break_returnStruct_call_super)
        self.assertEqual(
            len(threads), 1,
            "Stepped to the call super line in Source returnsStruct.")
        thread = threads[0]

        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(
            line_number, self.sourceBase_returnsStruct_start_line,
            "Stepped through super into SourceBase returnsStruct.")

        # Cool now continue to get past the call that initializes the Observer, and then do our steps in again to see that
        # we can find our way when we're stepping through a KVO swizzled
        # object.

        threads = lldbutil.continue_to_breakpoint(process, break3)
        self.assertEqual(
            len(threads), 1,
            "Continued to third breakpoint in main, our object should now be swizzled.")

        newClassName = mySource_isa.GetSummary()

        if self.TraceOn():
            print("className is %s, newClassName is %s" % (className, newClassName))
            print(mySource_isa)

        self.assertTrue(
            newClassName != className,
            "The isa did indeed change, swizzled!")

        # Now step in, that should leave us in the Source randomMethod:
        thread = threads[0]
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(
            line_number, self.source_randomMethod_line,
            "Stepped into Source randomMethod in swizzled object.")

        # Now step in again, through the super call, and that should leave us
        # in the SourceBase randomMethod:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(
            line_number, self.sourceBase_randomMethod_line,
            "Stepped through super into SourceBase randomMethod in swizzled object.")

        threads = lldbutil.continue_to_breakpoint(process, break4)
        self.assertEqual(
            len(threads), 1,
            "Continued to fourth breakpoint in main.")
        thread = threads[0]

        # Again, step in twice gets us to a stret method and a stret super
        # call:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(
            line_number, self.source_returnsStruct_start_line,
            "Stepped into Source returnsStruct in swizzled object.")

        threads = lldbutil.continue_to_breakpoint(
            process, break_returnStruct_call_super)
        self.assertEqual(
            len(threads), 1,
            "Stepped to the call super line in Source returnsStruct - second time.")
        thread = threads[0]

        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(
            line_number, self.sourceBase_returnsStruct_start_line,
            "Stepped through super into SourceBase returnsStruct in swizzled object.")

        for bkpt in breakpoints_to_disable:
            bkpt.SetEnabled(False)

        threads = lldbutil.continue_to_breakpoint(process, break_step_nil)
        self.assertEqual(len(threads), 1, "Continued to step nil breakpoint.")

        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(
            line_number, self.stepped_past_nil_line,
            "Step in over dispatch to nil stepped over.")
