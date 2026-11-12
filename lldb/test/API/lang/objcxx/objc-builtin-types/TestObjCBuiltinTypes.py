"""Test that the expression parser doesn't get confused by 'id' and 'Class'"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCBuiltinTypes(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.cpp"
        self.break_line = line_number(self.main_source, "// Set breakpoint here.")
        self.bar_break_line = line_number(self.main_source, "return id + Class")

    @add_test_categories(["pyapi"])
    def test_with_python_api(self):
        """Test expression parser respect for ObjC built-in types."""
        self.build()

        # Make sure we stopped at the first breakpoint.
        target, process, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec(self.main_source, False), self.break_line
        )

        bar_bpt = target.BreakpointCreateByLocation(
            self.main_source, self.bar_break_line
        )
        self.assertTrue(bar_bpt, VALID_BREAKPOINT)

        # Now make sure we can call a function in the class method we've
        # stopped in.
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

        self.expect("expr (foo)", patterns=[r"\(ns::id\) \$.* = 0"])

        self.expect(
            "expr --language Objective-C++ -- id my_id = 0; my_id",
            patterns=[r"\(id\) \$.* = nil"],
        )
        self.expect("expr --language C++ -- id my_id = 0; my_id", error=True)

        lldbutil.continue_to_breakpoint(process, bar_bpt)

        self.expect_expr("id", result_value="12", result_type="int")
        self.expect_expr("Class", result_value="15", result_type="int")
        self.expect("expr --language Objective-C++ -- id", error=True)
        self.expect("expr --language Objective-C++ -- Class", error=True)
