"""Test printing ObjC objects that use unbacked properties - so that the static ivar offsets are incorrect."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCIvarStripped(TestBase):
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.m"
        self.stop_line = line_number(self.main_source, "// Set breakpoint here.")

    @skipIf(
        debug_info=no_match("dsym"),
        bugnumber="This test requires a stripped binary and a dSYM",
    )
    @add_test_categories(["pyapi"])
    def test_with_python_api(self):
        """Test that we can find stripped Objective-C ivars in the runtime"""
        self.build()
        exe = self.getBuildArtifact("a.out.stripped")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.dbg.HandleCommand("add-dsym " + self.getBuildArtifact("a.out.dSYM"))

        breakpoint = target.BreakpointCreateByLocation(self.main_source, self.stop_line)
        self.assertTrue(
            breakpoint.IsValid() and breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT
        )

        _, _, thread, _ = lldbutil.run_to_breakpoint_do_run(self, target, breakpoint)

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame, "frame 0 is valid")

        # Test the expression for mc->_foo

        error = lldb.SBError()

        ivar = frame.EvaluateExpression("(mc->_foo)")
        self.assertTrue(ivar, "Got result for mc->_foo")
        ivar_value = ivar.GetValueAsSigned(error)
        self.assertSuccess(error)
        self.assertEqual(ivar_value, 3)
