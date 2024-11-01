"""
Test SBValue.GetObjectDescription() with the value from SBTarget.FindGlobalVariables().
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjectDescriptionAPITestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.source = "main.m"
        self.line = line_number(self.source, "// Set break point at this line.")

    # rdar://problem/10857337
    @add_test_categories(["pyapi"])
    def test_find_global_variables_then_object_description(self):
        """Exercise SBTarget.FindGlobalVariables() API."""
        d = {"EXE": "b.out"}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        exe = self.getBuildArtifact("b.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        # Make sure we hit our breakpoint:
        thread_list = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(len(thread_list), 1)

        thread = thread_list[0]
        frame0 = thread.GetFrameAtIndex(0)

        # Note my_global_str's object description prints fine here.
        value_list1 = frame0.GetVariables(True, True, True, True)
        for v in value_list1:
            self.DebugSBValue(v)
            if self.TraceOn():
                print("val:", v)
                print("object description:", v.GetObjectDescription())
            if v.GetName() == "my_global_str":
                self.assertEqual(v.GetObjectDescription(), "This is a global string")

        # But not here!
        value_list2 = target.FindGlobalVariables("my_global_str", 3)
        for v in value_list2:
            self.DebugSBValue(v)
            if self.TraceOn():
                print("val:", v)
                print("object description:", v.GetObjectDescription())
            if v.GetName() == "my_global_str":
                self.assertEqual(v.GetObjectDescription(), "This is a global string")
