"""
Test the WasHit feature of scripted breakpoints
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestWasHit(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24528")
    def test_was_hit_resolver(self):
        """Use facade breakpoints to emulate hitting some locations"""
        self.build()
        self.do_test()

    def make_target_and_import(self):
        target = lldbutil.run_to_breakpoint_make_target(self)
        self.import_resolver_script()
        return target

    def import_resolver_script(self):
        interp = self.dbg.GetCommandInterpreter()
        error = lldb.SBError()

        script_name = os.path.join(self.getSourceDir(), "bkpt_resolver.py")

        command = "command script import " + script_name
        self.runCmd(command)

    def make_extra_args(self, sym_name, num_locs, loc_to_miss):
        return f" -k symbol -v {sym_name} -k num_locs -v {num_locs} -k loc_to_miss -v {loc_to_miss} "

    def do_test(self):
        """This reads in a python file and sets a breakpoint using it."""

        target = self.make_target_and_import()
        extra_args = self.make_extra_args("stop_symbol", 4, 2)

        bkpt_no = lldbutil.run_break_set_by_script(
            self, "bkpt_resolver.FacadeExample", extra_args, 4
        )

        # Make sure the help text shows up in the "break list" output:
        self.expect(
            "break list",
            substrs=["I am a facade resolver - sym: stop_symbol - num_locs: 4"],
            msg="Help is listed in break list",
        )

        bkpt = target.FindBreakpointByID(bkpt_no)
        self.assertTrue(bkpt.IsValid(), "Found the right breakpoint")

        # Now continue.  We should hit locations 1, 3 and 4:
        (target, process, thread, bkpt) = lldbutil.run_to_breakpoint_do_run(
            self, target, bkpt
        )
        # This location should be bkpt_no.1:
        self.assertEqual(
            thread.stop_reason_data[0], bkpt_no, "Hit the right breakpoint"
        )
        self.assertEqual(thread.stop_reason_data[1], 1, "First location hit is 1")

        for loc in [3, 4]:
            process.Continue()
            self.assertEqual(
                thread.stop_reason, lldb.eStopReasonBreakpoint, "Hit breakpoint"
            )
            self.assertEqual(
                thread.stop_reason_data[0], bkpt_no, "Hit the right breakpoint"
            )
            self.assertEqual(
                thread.stop_reason_data[1], loc, f"Hit the right location: {loc}"
            )

        # At this point we should have hit three of the four locations, and not location 1.2.
        # Check that that is true, and that the descriptions for the location are the ones
        # the resolver provided.
        self.assertEqual(bkpt.hit_count, 3, "Hit three locations")
        for loc_id in range(1, 4):
            bkpt_loc = bkpt.FindLocationByID(loc_id)
            self.assertTrue(bkpt_loc.IsValid(), f"{loc_id} was invalid.")
            if loc_id != 2:
                self.assertEqual(
                    bkpt_loc.hit_count, 1, f"Loc {loc_id} hit count was wrong"
                )
            else:
                self.assertEqual(bkpt_loc.hit_count, 0, "We didn't skip loc 2")
            stream = lldb.SBStream()
            self.assertTrue(
                bkpt_loc.GetDescription(stream, lldb.eDescriptionLevelFull),
                f"Didn't get description for {loc_id}",
            )
            self.assertIn(
                f"Location index: {loc_id}",
                stream.GetData(),
                f"Wrong desciption for {loc_id}",
            )
