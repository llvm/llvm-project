"""
Test setting breakpoints using a scripted resolver
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestScriptedResolver(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24528")
    def test_scripted_resolver(self):
        """Use a scripted resolver to set a by symbol name breakpoint"""
        self.build()
        self.do_test()

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24528")
    def test_search_depths(self):
        """ Make sure we are called at the right depths depending on what we return
            from __get_depth__"""
        self.build()
        self.do_test_depths()

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24528")
    def test_command_line(self):
        """ Test setting a resolver breakpoint from the command line """
        self.build()
        self.do_test_cli()

    def test_bad_command_lines(self):
        """Make sure we get appropriate errors when we give invalid key/value
           options"""
        self.build()
        self.do_test_bad_options()

    def test_copy_from_dummy_target(self):
        """Make sure we don't crash during scripted breakpoint copy from dummy target"""
        self.build()
        self.do_test_copy_from_dummy_target()

    def make_target_and_import(self):
        target = self.make_target()
        self.import_resolver_script()
        return target

    def make_target(self):
        return lldbutil.run_to_breakpoint_make_target(self)

    def import_resolver_script(self):
        interp = self.dbg.GetCommandInterpreter()
        error = lldb.SBError()

        script_name = os.path.join(self.getSourceDir(), "resolver.py")
        source_name = os.path.join(self.getSourceDir(), "main.c")

        command = "command script import " + script_name
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand(command, result)
        self.assertTrue(result.Succeeded(), "com scr imp failed: %s"%(result.GetError()))

    def make_extra_args(self):
        json_string = '{"symbol":"break_on_me", "test1": "value1"}'
        json_stream = lldb.SBStream()
        json_stream.Print(json_string)
        extra_args = lldb.SBStructuredData()
        error = extra_args.SetFromJSON(json_stream)
        self.assertSuccess(error, "Error making SBStructuredData")
        return extra_args

    def do_test(self):
        """This reads in a python file and sets a breakpoint using it."""

        target = self.make_target_and_import()
        extra_args = self.make_extra_args()

        file_list = lldb.SBFileSpecList()
        module_list = lldb.SBFileSpecList()

        # Make breakpoints with this resolver using different filters, first ones that will take:
        right = []
        # one with no file or module spec - this one should fire:
        right.append(target.BreakpointCreateFromScript("resolver.Resolver", extra_args, module_list, file_list))

        # one with the right source file and no module - should also fire:
        file_list.Append(lldb.SBFileSpec("main.c"))
        right.append(target.BreakpointCreateFromScript("resolver.Resolver", extra_args, module_list, file_list))
        # Make sure the help text shows up in the "break list" output:
        self.expect("break list", substrs=["I am a python breakpoint resolver"], msg="Help is listed in break list")

        # one with the right source file and right module - should also fire:
        module_list.Append(lldb.SBFileSpec("a.out"))
        right.append(target.BreakpointCreateFromScript("resolver.Resolver", extra_args, module_list, file_list))

        # And one with no source file but the right module:
        file_list.Clear()
        right.append(target.BreakpointCreateFromScript("resolver.Resolver", extra_args, module_list, file_list))

        # Make sure these all got locations:
        for i in range (0, len(right)):
            self.assertTrue(right[i].GetNumLocations() >= 1, "Breakpoint %d has no locations."%(i))

        # Now some ones that won't take:

        module_list.Clear()
        file_list.Clear()
        wrong = []

        # one with the wrong module - should not fire:
        module_list.Append(lldb.SBFileSpec("noSuchModule"))
        wrong.append(target.BreakpointCreateFromScript("resolver.Resolver", extra_args, module_list, file_list))

        # one with the wrong file - also should not fire:
        file_list.Clear()
        module_list.Clear()
        file_list.Append(lldb.SBFileSpec("noFileOfThisName.xxx"))
        wrong.append(target.BreakpointCreateFromScript("resolver.Resolver", extra_args, module_list, file_list))

        # Now make sure the CU level iteration obeys the file filters:
        file_list.Clear()
        module_list.Clear()
        file_list.Append(lldb.SBFileSpec("no_such_file.xxx"))
        wrong.append(target.BreakpointCreateFromScript("resolver.ResolverCUDepth", extra_args, module_list, file_list))

        # And the Module filters:
        file_list.Clear()
        module_list.Clear()
        module_list.Append(lldb.SBFileSpec("NoSuchModule.dylib"))
        wrong.append(target.BreakpointCreateFromScript("resolver.ResolverCUDepth", extra_args, module_list, file_list))

        # Now make sure the Function level iteration obeys the file filters:
        file_list.Clear()
        module_list.Clear()
        file_list.Append(lldb.SBFileSpec("no_such_file.xxx"))
        wrong.append(target.BreakpointCreateFromScript("resolver.ResolverFuncDepth", extra_args, module_list, file_list))

        # And the Module filters:
        file_list.Clear()
        module_list.Clear()
        module_list.Append(lldb.SBFileSpec("NoSuchModule.dylib"))
        wrong.append(target.BreakpointCreateFromScript("resolver.ResolverFuncDepth", extra_args, module_list, file_list))

        # Make sure these didn't get locations:
        for i in range(0, len(wrong)):
            self.assertEqual(wrong[i].GetNumLocations(), 0, "Breakpoint %d has locations."%(i))

        # Now run to main and ensure we hit the breakpoints we should have:

        lldbutil.run_to_breakpoint_do_run(self, target, right[0])

        # Test the hit counts:
        for i in range(0, len(right)):
            self.assertEqual(right[i].GetHitCount(), 1, "Breakpoint %d has the wrong hit count"%(i))

        for i in range(0, len(wrong)):
            self.assertEqual(wrong[i].GetHitCount(), 0, "Breakpoint %d has the wrong hit count"%(i))

    def do_test_depths(self):
        """This test uses a class variable in resolver.Resolver which gets set to 1 if we saw
           compile unit and 2 if we only saw modules.  If the search depth is module, you get passed just
           the modules with no comp_unit.  If the depth is comp_unit you get comp_units.  So we can use
           this to test that our callback gets called at the right depth."""

        target = self.make_target_and_import()
        extra_args = self.make_extra_args()

        file_list = lldb.SBFileSpecList()
        module_list = lldb.SBFileSpecList()
        module_list.Append(lldb.SBFileSpec("a.out"))

        # Make a breakpoint that has no __get_depth__, check that that is converted to eSearchDepthModule:
        bkpt = target.BreakpointCreateFromScript("resolver.Resolver", extra_args, module_list, file_list)
        self.assertTrue(bkpt.GetNumLocations() > 0, "Resolver got no locations.")
        self.expect("script print(resolver.Resolver.got_files)", substrs=["2"], msg="Was only passed modules")

        # Make a breakpoint that asks for modules, check that we didn't get any files:
        bkpt = target.BreakpointCreateFromScript("resolver.ResolverModuleDepth", extra_args, module_list, file_list)
        self.assertTrue(bkpt.GetNumLocations() > 0, "ResolverModuleDepth got no locations.")
        self.expect("script print(resolver.Resolver.got_files)", substrs=["2"], msg="Was only passed modules")

        # Make a breakpoint that asks for compile units, check that we didn't get any files:
        bkpt = target.BreakpointCreateFromScript("resolver.ResolverCUDepth", extra_args, module_list, file_list)
        self.assertTrue(bkpt.GetNumLocations() > 0, "ResolverCUDepth got no locations.")
        self.expect("script print(resolver.Resolver.got_files)", substrs=["1"], msg="Was passed compile units")

        # Make a breakpoint that returns a bad value - we should convert that to "modules" so check that:
        bkpt = target.BreakpointCreateFromScript("resolver.ResolverBadDepth", extra_args, module_list, file_list)
        self.assertTrue(bkpt.GetNumLocations() > 0, "ResolverBadDepth got no locations.")
        self.expect("script print(resolver.Resolver.got_files)", substrs=["2"], msg="Was only passed modules")

        # Make a breakpoint that searches at function depth:
        bkpt = target.BreakpointCreateFromScript("resolver.ResolverFuncDepth", extra_args, module_list, file_list)
        self.assertTrue(bkpt.GetNumLocations() > 0, "ResolverFuncDepth got no locations.")
        self.expect("script print(resolver.Resolver.got_files)", substrs=["3"], msg="Was only passed modules")
        self.expect("script print(resolver.Resolver.func_list)", substrs=['test_func', 'break_on_me', 'main'], msg="Saw all the functions")

    def do_test_cli(self):
        target = self.make_target_and_import()

        lldbutil.run_break_set_by_script(self, "resolver.Resolver", extra_options="-k symbol -v break_on_me")

        # Make sure setting a resolver breakpoint doesn't pollute further breakpoint setting
        # by checking the description of a regular file & line breakpoint to make sure it
        # doesn't mention the Python Resolver function:
        bkpt_no = lldbutil.run_break_set_by_file_and_line(self, "main.c", 12)
        bkpt = target.FindBreakpointByID(bkpt_no)
        strm = lldb.SBStream()
        bkpt.GetDescription(strm, False)
        used_resolver = "I am a python breakpoint resolver" in strm.GetData()
        self.assertFalse(used_resolver, "Found the resolver description in the file & line breakpoint description.")

        # Also make sure the breakpoint was where we expected:
        bp_loc = bkpt.GetLocationAtIndex(0)
        bp_sc = bp_loc.GetAddress().GetSymbolContext(lldb.eSymbolContextEverything)
        bp_se = bp_sc.GetLineEntry()
        self.assertEqual(bp_se.GetLine(), 12, "Got the right line number")
        self.assertEqual(bp_se.GetFileSpec().GetFilename(), "main.c", "Got the right filename")

    def do_test_bad_options(self):
        target = self.make_target_and_import()

        self.expect("break set -P resolver.Resolver -k a_key", error = True, msg="Missing value at end",
           substrs=['Key: "a_key" missing value'])
        self.expect("break set -P resolver.Resolver -v a_value", error = True, msg="Missing key at end",
           substrs=['Value: "a_value" missing matching key'])
        self.expect("break set -P resolver.Resolver -v a_value -k a_key -v another_value", error = True, msg="Missing key among args",
           substrs=['Value: "a_value" missing matching key'])
        self.expect("break set -P resolver.Resolver -k a_key -k a_key -v another_value", error = True, msg="Missing value among args",
           substrs=['Key: "a_key" missing value'])

    def do_test_copy_from_dummy_target(self):
        # Import breakpoint scripted resolver.
        self.import_resolver_script()

        # Create a scripted breakpoint.
        self.runCmd("breakpoint set -P resolver.Resolver -k symbol -v break_on_me",
                    BREAKPOINT_CREATED)

        # This is the function to remove breakpoints from the dummy target
        # to get a clean state for the next test case.
        def cleanup():
            self.runCmd('breakpoint delete -D -f', check=False)
            self.runCmd('breakpoint list', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Check that target creating doesn't crash.
        target = self.make_target()
