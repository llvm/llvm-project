"""
Test that you can set breakpoint commands successfully with the Python API's:
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import side_effect


class PythonBreakpointCommandSettingTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(['pyapi'])
    def test_step_out_python(self):
        """Test stepping out using a python breakpoint command."""
        self.build()
        self.do_set_python_command_from_python()

    def test_bkpt_cmd_bad_arguments(self):
        """Test what happens when pass structured data to a command:"""
        self.build()
        self.do_bad_args_to_python_command()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.c"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_set_python_command_from_python(self):
        error = lldb.SBError()

        self.target = self.createTestTarget()

        body_bkpt = self.target.BreakpointCreateBySourceRegex(
            "Set break point at this line.", self.main_source_spec)
        self.assertTrue(body_bkpt, VALID_BREAKPOINT)

        func_bkpt = self.target.BreakpointCreateBySourceRegex(
            "Set break point at this line.", self.main_source_spec)
        self.assertTrue(func_bkpt, VALID_BREAKPOINT)

        fancy_bkpt = self.target.BreakpointCreateBySourceRegex(
            "Set break point at this line.", self.main_source_spec)
        self.assertTrue(fancy_bkpt, VALID_BREAKPOINT)

        fancier_bkpt = self.target.BreakpointCreateBySourceRegex(
            "Set break point at this line.", self.main_source_spec)
        self.assertTrue(fancier_bkpt, VALID_BREAKPOINT)

        # Also test the list version of this:
        file_list = lldb.SBFileSpecList()
        file_list.Append(self.main_source_spec)
        module_list = lldb.SBFileSpecList()
        module_list.Append(self.target.GetExecutable())
        
        list_bkpt = self.target.BreakpointCreateBySourceRegex(
            "Set break point at this line.", module_list, file_list)
        self.assertTrue(list_bkpt, VALID_BREAKPOINT)

        
        not_so_fancy_bkpt = self.target.BreakpointCreateBySourceRegex(
            "Set break point at this line.", self.main_source_spec)
        self.assertTrue(not_so_fancy_bkpt, VALID_BREAKPOINT)

        # Also test that setting a source regex breakpoint with an empty file
        # spec list sets it on all files:
        no_files_bkpt = self.target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpecList(), lldb.SBFileSpecList())
        self.assertTrue(no_files_bkpt, VALID_BREAKPOINT)
        num_locations = no_files_bkpt.GetNumLocations()
        self.assertTrue(
            num_locations >= 2,
            "Got at least two breakpoint locations")
        got_one_in_A = False
        got_one_in_B = False
        for idx in range(0, num_locations):
            comp_unit = no_files_bkpt.GetLocationAtIndex(idx).GetAddress().GetSymbolContext(
                lldb.eSymbolContextCompUnit).GetCompileUnit().GetFileSpec()
            print("Got comp unit: ", comp_unit.GetFilename())
            if comp_unit.GetFilename() == "a.c":
                got_one_in_A = True
            elif comp_unit.GetFilename() == "b.c":
                got_one_in_B = True

        self.assertTrue(got_one_in_A, "Failed to match the pattern in A")
        self.assertTrue(got_one_in_B, "Failed to match the pattern in B")
        self.target.BreakpointDelete(no_files_bkpt.GetID())

        error = lldb.SBError()
        error = body_bkpt.SetScriptCallbackBody(
                "import side_effect; side_effect.callback = 'callback was here'")
        self.assertTrue(
            error.Success(),
            "Failed to set the script callback body: %s." %
            (error.GetCString()))

        self.expect("command script import --allow-reload ./bktptcmd.py")

        func_bkpt.SetScriptCallbackFunction("bktptcmd.function")

        extra_args = lldb.SBStructuredData()
        stream = lldb.SBStream()
        stream.Print('{"side_effect" : "I am fancy"}')
        extra_args.SetFromJSON(stream)
        error = fancy_bkpt.SetScriptCallbackFunction("bktptcmd.another_function", extra_args)
        self.assertSuccess(error, "Failed to add callback")

        stream.Clear()
        stream.Print('{"side_effect" : "I am so much fancier"}')
        extra_args.SetFromJSON(stream)

        # Fancier's callback is set up from the command line
        id = fancier_bkpt.GetID()
        self.expect("breakpoint command add -F bktptcmd.a_third_function -k side_effect -v 'I am fancier' %d"%(id))

        # Not so fancy gets an empty extra_args:
        empty_args = lldb.SBStructuredData()
        error = not_so_fancy_bkpt.SetScriptCallbackFunction("bktptcmd.empty_extra_args", empty_args)
        self.assertSuccess(error, "Failed to add callback")

        # Do list breakpoint like fancy:
        stream.Clear()
        stream.Print('{"side_effect" : "I come from list input"}')
        extra_args.SetFromJSON(stream)
        error = list_bkpt.SetScriptCallbackFunction("bktptcmd.a_list_function", extra_args)
        self.assertSuccess(error, "Failed to add callback")
        
        # Clear out canary variables
        side_effect.bktptcmd = None
        side_effect.callback = None
        side_effect.fancy    = None
        side_effect.fancier  = None
        side_effect.not_so_fancy = None
        side_effect.a_list_function = None
        
        # Now launch the process, and do not stop at entry point.
        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(self.process, PROCESS_IS_VALID)

        # Now finish, and make sure the return value is correct.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, body_bkpt)
        self.assertEquals(len(threads), 1, "Stopped at inner breakpoint.")
        self.thread = threads[0]

        print("* Num Locations: {0} ; Hit Count {1}".format(list_bkpt.GetNumLocations(), list_bkpt.GetHitCount()))
        self.assertEquals("callback was here", side_effect.callback)
        self.assertEquals("function was here", side_effect.bktptcmd)
        self.assertEquals("I am fancy", side_effect.fancy)
        self.assertEquals("I am fancier", side_effect.fancier)
        self.assertEquals("Not so fancy", side_effect.not_so_fancy)
        self.assertEquals("I come from list input", side_effect.from_list)

    def do_bad_args_to_python_command(self):
        error = lldb.SBError()

        self.target = self.createTestTarget()

        self.expect("command script import --allow-reload ./bktptcmd.py")

        bkpt = self.target.BreakpointCreateBySourceRegex(
            "Set break point at this line.", self.main_source_spec)
        self.assertTrue(bkpt, VALID_BREAKPOINT)

        # Pass a breakpoint command function that doesn't take extra_args,
        # but pass it extra args:

        extra_args = lldb.SBStructuredData()
        stream = lldb.SBStream()
        stream.Print('{"side_effect" : "I am fancy"}')
        extra_args.SetFromJSON(stream)

        error = bkpt.SetScriptCallbackFunction("bktptcmd.function", extra_args)
        self.assertTrue(error.Fail(), "Can't pass extra args if the function doesn't take them")

        error = bkpt.SetScriptCallbackFunction("bktptcmd.useless_function", extra_args)
        self.assertTrue(error.Fail(), "Can't pass extra args if the function has wrong number of args.")

        error = bkpt.SetScriptCallbackFunction("bktptcmd.nosuch_function", extra_args)
        self.assertTrue(error.Fail(), "Can't pass extra args if the function doesn't exist.")

