"""
Test the "process continue -b" option.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestContinueToBkpts(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(["pyapi"])
    def test_continue_to_breakpoints(self):
        """Test that the continue to breakpoints feature works correctly."""
        self.build()
        self.do_test_continue_to_breakpoint()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.main_source_spec = lldb.SBFileSpec("main.c")

    def continue_and_check(self, stop_list, bkpt_to_hit, loc_to_hit=0):
        """Build up a command that will run a continue -b commands using the breakpoints on stop_list, and
        ensure that we hit bkpt_to_hit.
        If loc_to_hit is not 0, also verify that we hit that location."""
        command = "process continue"
        for elem in stop_list:
            command += " -b {0}".format(elem)
        self.expect(command)
        self.assertStopReason(
            self.thread.stop_reason, lldb.eStopReasonBreakpoint, "Hit a breakpoint"
        )
        self.assertEqual(
            self.thread.GetStopReasonDataAtIndex(0),
            bkpt_to_hit,
            "Hit the right breakpoint",
        )
        if loc_to_hit != 0:
            self.assertEqual(
                self.thread.GetStopReasonDataAtIndex(1),
                loc_to_hit,
                "Hit the right location",
            )
        for bkpt_id in self.bkpt_list:
            bkpt = self.target.FindBreakpointByID(bkpt_id)
            self.assertTrue(bkpt.IsValid(), "Breakpoint id's round trip")
            if bkpt.MatchesName("disabled"):
                self.assertFalse(
                    bkpt.IsEnabled(),
                    "Disabled breakpoints stay disabled: {0}".format(bkpt.GetID()),
                )
            else:
                self.assertTrue(
                    bkpt.IsEnabled(),
                    "Enabled breakpoints stay enabled: {0}".format(bkpt.GetID()),
                )
        # Also do our multiple location one:
        bkpt = self.target.FindBreakpointByID(self.multiple_loc_id)
        self.assertTrue(bkpt.IsValid(), "Breakpoint with locations round trip")
        for i in range(1, 3):
            loc = bkpt.FindLocationByID(i)
            self.assertTrue(loc.IsValid(), "Locations round trip")
            if i == 2:
                self.assertTrue(
                    loc.IsEnabled(), "Locations that were enabled stay enabled"
                )
            else:
                self.assertFalse(
                    loc.IsEnabled(), "Locations that were disabled stay disabled"
                )

    def do_test_continue_to_breakpoint(self):
        """Test the continue to breakpoint feature."""
        (self.target, process, self.thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Stop here to get started", self.main_source_spec
        )

        # Now set up all our breakpoints:
        bkpt_pattern = "This is the {0} stop"
        bkpt_elements = [
            "zeroth",
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "nineth",
        ]
        disabled_bkpts = ["first", "eigth"]
        bkpts_for_MyBKPT = ["first", "sixth", "nineth"]
        self.bkpt_list = []
        for elem in bkpt_elements:
            bkpt = self.target.BreakpointCreateBySourceRegex(
                bkpt_pattern.format(elem), self.main_source_spec
            )
            self.assertGreater(bkpt.GetNumLocations(), 0, "Found a bkpt match")
            self.bkpt_list.append(bkpt.GetID())
            bkpt.AddName(elem)
            if elem in disabled_bkpts:
                bkpt.AddName("disabled")
                bkpt.SetEnabled(False)
            if elem in bkpts_for_MyBKPT:
                bkpt.AddName("MyBKPT")
        # Also make one that has several locations, so we can test locations:
        mult_bkpt = self.target.BreakpointCreateBySourceRegex(
            bkpt_pattern.format("(seventh|eighth|nineth)"), self.main_source_spec
        )
        self.assertEqual(mult_bkpt.GetNumLocations(), 3, "Got three matches")
        mult_bkpt.AddName("Locations")
        # Disable all of these:
        for i in range(1, 4):
            loc = mult_bkpt.FindLocationByID(i)
            self.assertTrue(loc.IsValid(), "Location {0} is valid".format(i))
            loc.SetEnabled(False)
            self.assertFalse(loc.IsEnabled(), "Loc {0} wasn't disabled".format(i))
        self.multiple_loc_id = mult_bkpt.GetID()

        # First test out various error conditions

        # All locations of the multiple_loc_id are disabled, so running to this should be an error:
        self.expect(
            "process continue -b {0}".format(self.multiple_loc_id),
            error=True,
            msg="Running to a disabled breakpoint by number",
        )

        # Now re-enable the middle one so we can run to it:
        loc = mult_bkpt.FindLocationByID(2)
        loc.SetEnabled(True)

        self.expect(
            "process continue -b {0}".format(self.bkpt_list[1]),
            error=True,
            msg="Running to a disabled breakpoint by number",
        )
        self.expect(
            "process continue -b {0}.1".format(self.bkpt_list[1]),
            error=True,
            msg="Running to a location of a disabled breakpoint",
        )
        self.expect(
            "process continue -b disabled",
            error=True,
            msg="Running to a disabled set of breakpoints",
        )
        self.expect(
            "process continue -b {0}.{1}".format(self.multiple_loc_id, 1),
            error=True,
            msg="Running to a disabled breakpoint location",
        )
        self.expect(
            "process continue -b {0}".format("THERE_ARE_NO_BREAKPOINTS_BY_THIS_NAME"),
            error=True,
            msg="Running to no such name",
        )
        self.expect(
            "process continue -b {0}".format(1000),
            error=True,
            msg="Running to no such breakpoint",
        )
        self.expect(
            "process continue -b {0}.{1}".format(self.multiple_loc_id, 1000),
            error=True,
            msg="Running to no such location",
        )

        # Now move forward, this time with breakpoint numbers.  First time we don't skip other bkpts.
        bkpt = self.bkpt_list[0]
        self.continue_and_check([str(bkpt)], bkpt)

        # Now skip to the third stop, do it by name and supply one of the later breakpoints as well:
        # This continue has to muck with the sync mode of the debugger, so let's make sure we
        # put it back.  First try if it was in sync mode:
        orig_async = self.dbg.GetAsync()
        self.dbg.SetAsync(True)
        self.continue_and_check([bkpt_elements[2], bkpt_elements[7]], self.bkpt_list[2])
        after_value = self.dbg.GetAsync()
        self.dbg.SetAsync(orig_async)
        self.assertTrue(after_value, "Preserve async as True if it started that way")

        # Now try a name that has several breakpoints.
        # This time I'm also going to check that we put the debugger async mode back if
        # if was False to begin with:
        self.dbg.SetAsync(False)
        self.continue_and_check(["MyBKPT"], self.bkpt_list[6])
        after_value = self.dbg.GetAsync()
        self.dbg.SetAsync(orig_async)
        self.assertFalse(after_value, "Preserve async as False if it started that way")

        # Now let's run to a particular location.  Also specify a breakpoint we've already hit:
        self.continue_and_check(
            [self.bkpt_list[0], self.multiple_loc_id], self.multiple_loc_id, 2
        )
