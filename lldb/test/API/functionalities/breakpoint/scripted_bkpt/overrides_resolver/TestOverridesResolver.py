"""
Test the OverridesResolver feature of scripted breakpoints
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestOverridesResolver(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24528")
    def test_overrides_resolver_resolver_python(self):
        """Use facade breakpoints to emulate hitting some locations"""
        self.build()
        self.do_test(True)

    def test_overrides_resolver_resolver_cmd(self):
        """Use facade breakpoints to emulate hitting some locations"""
        self.build()
        self.do_test(True)

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

    def do_test(self, use_cmd):
        """This reads in a python file and sets a breakpoint using it."""
        alternate_location = "stop_here_instead"
        target = self.make_target_and_import()
        help_text = "SOME HELP TEXT"
        class_name = "bkpt_resolver.OverrideExample"
        key = "symbol"
        value = "stop_here_instead"

        if use_cmd:
            result = lldb.SBCommandReturnObject()
            self.ci.HandleCommand(
                f"breakpoint override add -P {class_name} -k {key} -v {value} -d '{help_text}'",
                result,
            )
            self.assertCommandReturn(result, "breakpoint override worked")
            override_id = int(result.GetOutput())
        else:
            extra_args = lldb.SBStructuredData()
            json_str = '{"' + key + '":"' + value + '"}'
            extra_args.SetFromJSON(json_str)
            override_id = target.AddBreakpointOverride(
                class_name, help_text, extra_args
            )

        # Check the override listing, make sure our new entry is present:
        self.expect("breakpoint override list", substrs=[str(override_id), help_text])

        # Now make a breakpoint by file and line:
        # FIXME: Use source_line to find this line number:
        bkpt = target.BreakpointCreateByLocation(
            "main.c", line_number("main.c", "I am in the stop symbol")
        )
        self.assertEqual(bkpt.GetNumLocations(), 1, "We make one location")
        # Now continue and we'll hit this breakpoint but not in the
        # right place:
        (target, process, thread, bkpt) = lldbutil.run_to_breakpoint_do_run(
            self, target, bkpt
        )
        # This location should be bkpt_no.1:
        self.assertEqual(
            thread.stop_reason_data[0], bkpt.GetID(), "Hit the right breakpoint"
        )
        self.assertEqual(thread.stop_reason_data[1], 1, "First location hit is 1")
        func_name = thread.frames[0].name
        self.assertEqual(
            func_name, alternate_location, "Stopped at overridden location"
        )

        # Now set a source name breakpoint, that should not get overridden, and
        # when we continue we should hit it:
        name_bkpt = target.BreakpointCreateByName("change_him")
        self.assertGreater(name_bkpt.GetNumLocations(), 0, "Found locations")
        threads = lldbutil.continue_to_breakpoint(process, name_bkpt)
        self.assertEqual(len(threads), 1, "Hit our name breakpoint")
        func_name = threads[0].frames[0].name
        self.assertEqual(func_name, "change_him", "Stopped in the right place")

        # Now delete the override and make sure we hit newly set
        # source breakpoints:
        if use_cmd:
            self.runCmd(f"breakpoint override delete {override_id}")
        else:
            self.assertTrue(
                target.DeleteBreakpointOverride(override_id), "Delete the right one"
            )

        # FIXME use source_line:
        new_bkpt = target.BreakpointCreateByLocation(
            "main.c", line_number("main.c", "return 0")
        )
        self.assertEqual(new_bkpt.num_locations, 1, "Made breakpoint")
        threads = lldbutil.continue_to_breakpoint(process, new_bkpt)
        self.assertEqual(len(threads), 1, "Hit our new breakpoint")
        func_name = threads[0].frames[0].name
        self.assertEqual(func_name, "main", "Stopped in unchanged location")
