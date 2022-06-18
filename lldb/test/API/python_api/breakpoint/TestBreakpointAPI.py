"""
Test SBBreakpoint APIs.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointAPITestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_breakpoint_is_valid(self):
        """Make sure that if an SBBreakpoint gets deleted its IsValid returns false."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'AFunction'.
        breakpoint = target.BreakpointCreateByName('AFunction', 'a.out')
        self.trace("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now delete it:
        did_delete = target.BreakpointDelete(breakpoint.GetID())
        self.assertTrue(
            did_delete,
            "Did delete the breakpoint we just created.")

        # Make sure we can't find it:
        del_bkpt = target.FindBreakpointByID(breakpoint.GetID())
        self.assertTrue(not del_bkpt, "We did delete the breakpoint.")

        # Finally make sure the original breakpoint is no longer valid.
        self.assertTrue(
            not breakpoint,
            "Breakpoint we deleted is no longer valid.")

    def test_target_delete(self):
        """Make sure that if an SBTarget gets deleted the associated
        Breakpoint's IsValid returns false."""

        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'AFunction'.
        breakpoint = target.BreakpointCreateByName('AFunction', 'a.out')
        self.trace("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location.IsValid())

        # Make sure the breakpoint's target is right:
        self.assertEqual(target, breakpoint.GetTarget(), "Breakpoint reports its target correctly")
        
        self.assertTrue(self.dbg.DeleteTarget(target))
        self.assertFalse(breakpoint.IsValid())
        self.assertFalse(location.IsValid())
