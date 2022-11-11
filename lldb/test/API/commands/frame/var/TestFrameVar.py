"""
Make sure the frame variable -g, -a, and -l flags work.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import os
import shutil
import time

class TestFrameVar(TestBase):

    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        self.do_test()

    def do_test(self):
        target = self.createTestTarget()

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.c"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() >= 1,
                        VALID_BREAKPOINT)

        error = lldb.SBError()
        # This is the launch info.  If you want to launch with arguments or
        # environment variables, add them using SetArguments or
        # SetEnvironmentEntries

        launch_info = target.GetLaunchInfo()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Did we hit our breakpoint?
        from lldbsuite.test.lldbutil import get_threads_stopped_at_breakpoint
        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(
            len(threads), 1,
            "There should be a thread stopped at our breakpoint")

        # The hit count for the breakpoint should be 1.
        self.assertEquals(breakpoint.GetHitCount(), 1)

        frame = threads[0].GetFrameAtIndex(0)
        command_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()

        # Just get args:
        result = interp.HandleCommand("frame var -l", command_result)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "frame var -a didn't succeed")
        output = command_result.GetOutput()
        self.assertIn("argc", output, "Args didn't find argc")
        self.assertIn("argv", output, "Args didn't find argv")
        self.assertNotIn("test_var", output, "Args found a local")
        self.assertNotIn("g_var", output, "Args found a global")

        # Just get locals:
        result = interp.HandleCommand("frame var -a", command_result)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "frame var -a didn't succeed")
        output = command_result.GetOutput()
        self.assertNotIn("argc", output, "Locals found argc")
        self.assertNotIn("argv", output, "Locals found argv")
        self.assertIn("test_var", output, "Locals didn't find test_var")
        self.assertNotIn("g_var", output, "Locals found a global")

        # Get the file statics:
        result = interp.HandleCommand("frame var -l -a -g", command_result)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "frame var -a didn't succeed")
        output = command_result.GetOutput()
        self.assertNotIn("argc", output, "Globals found argc")
        self.assertNotIn("argv", output, "Globals found argv")
        self.assertNotIn("test_var", output, "Globals found test_var")
        self.assertIn("g_var", output, "Globals didn't find g_var")


    def check_frame_variable_errors(self, thread, error_strings):
        command_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()
        result = interp.HandleCommand("frame variable", command_result)
        self.assertEqual(result, lldb.eReturnStatusFailed,
                         "frame var succeeded unexpectedly")
        command_error = command_result.GetError()

        frame = thread.GetFrameAtIndex(0)
        var_list = frame.GetVariables(True, True, False, True)
        self.assertEqual(var_list.GetSize(), 0)
        api_error = var_list.GetError().GetCString()

        for s in error_strings:
            self.assertIn(s, command_error)
        for s in error_strings:
            self.assertIn(s, api_error)


    @skipIfRemote
    @skipUnlessDarwin
    def test_darwin_dwarf_missing_obj(self):
        '''
            Test that if we build a binary with DWARF in .o files and we remove
            the .o file for main.cpp, that we get an appropriate error when we
            do 'frame variable' that explains why we aren't seeing variables.
        '''
        self.build(debug_info="dwarf")
        exe = self.getBuildArtifact("a.out")
        main_obj = self.getBuildArtifact("main.o")
        # Delete the main.o file that contains the debug info so we force an
        # error when we run to main and try to get variables
        os.unlink(main_obj)

        # We have to set a named breakpoint because we don't have any debug info
        # because we deleted the main.o file.
        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(self, 'main')
        error_strings = [
            'debug map object file "',
            'main.o" containing debug info does not exist, debug info will not be loaded'
        ]
        self.check_frame_variable_errors(thread, error_strings)


    @skipIfRemote
    @skipUnlessDarwin
    def test_darwin_dwarf_obj_mod_time_mismatch(self):
        '''
            Test that if we build a binary with DWARF in .o files and we update
            the mod time of the .o file for main.cpp, that we get an appropriate
            error when we do 'frame variable' that explains why we aren't seeing
            variables.
        '''
        self.build(debug_info="dwarf")
        exe = self.getBuildArtifact("a.out")
        main_obj = self.getBuildArtifact("main.o")

        # Set the modification time for main.o file to the current time after
        # sleeping for 2 seconds. This ensures the modification time will have
        # changed and will not match the modification time in the debug map and
        # force an error when we run to main and try to get variables
        time.sleep(2)
        os.utime(main_obj, None)

        # We have to set a named breakpoint because we don't have any debug info
        # because we deleted the main.o file since the mod times don't match
        # and debug info won't be loaded
        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(self, 'main')

        error_strings = [
            'debug map object file "',
            'main.o" changed (actual: 0x',
            ', debug map: 0x',
            ') since this executable was linked, debug info will not be loaded'
        ]
        self.check_frame_variable_errors(thread, error_strings)


    @skipIfRemote
    @skipIfWindows # Windows can't set breakpoints by name 'main' in this case.
    def test_gline_tables_only(self):
        '''
            Test that if we build a binary with "-gline-tables-only" that we can
            set a file and line breakpoint successfully, and get an error
            letting us know that this build option was enabled when trying to
            read variables.
        '''
        self.build(dictionary={'CFLAGS_EXTRAS': '-gline-tables-only'})
        exe = self.getBuildArtifact("a.out")

        # We have to set a named breakpoint because we don't have any debug info
        # because we deleted the main.o file since the mod times don't match
        # and debug info won't be loaded
        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(self, 'main')
        error_strings = [
            'no variable information is available in debug info for this compile unit'
        ]
        self.check_frame_variable_errors(thread, error_strings)

    @skipUnlessPlatform(["linux", "freebsd"])
    @add_test_categories(["dwo"])
    def test_fission_missing_dwo(self):
        '''
            Test that if we build a binary with "-gsplit-dwarf" that we can
            set a file and line breakpoint successfully, and get an error
            letting us know we were unable to load the .dwo file.
        '''
        self.build(dictionary={'CFLAGS_EXTRAS': '-gsplit-dwarf'})
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("main.dwo")

        self.assertTrue(os.path.exists(main_dwo), 'Make sure "%s" file exists' % (main_dwo))
        # Delete the main.dwo file that contains the debug info so we force an
        # error when we run to main and try to get variables.
        os.unlink(main_dwo)

        # We have to set a named breakpoint because we don't have any debug info
        # because we deleted the main.o file since the mod times don't match
        # and debug info won't be loaded
        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(self, 'main')
        error_strings = [
            'unable to locate .dwo debug file "',
            'main.dwo" for skeleton DIE 0x'
        ]
        self.check_frame_variable_errors(thread, error_strings)

    @skipUnlessPlatform(["linux", "freebsd"])
    @add_test_categories(["dwo"])
    def test_fission_invalid_dwo_objectfile(self):
        '''
            Test that if we build a binary with "-gsplit-dwarf" that we can
            set a file and line breakpoint successfully, and get an error
            letting us know we were unable to load the .dwo file because it
            existed, but it wasn't a valid object file.
        '''
        self.build(dictionary={'CFLAGS_EXTRAS': '-gsplit-dwarf'})
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("main.dwo")

        self.assertTrue(os.path.exists(main_dwo), 'Make sure "%s" file exists' % (main_dwo))
        # Overwrite the main.dwo with the main.c source file so that the .dwo
        # file exists, but it isn't a valid object file as there is an error
        # for this case.
        shutil.copyfile(self.getSourcePath('main.c'), main_dwo)

        # We have to set a named breakpoint because we don't have any debug info
        # because we deleted the main.o file since the mod times don't match
        # and debug info won't be loaded
        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(self, 'main')
        error_strings = [
            'unable to load object file for .dwo debug file "'
            'main.dwo" for unit DIE 0x',
        ]
        self.check_frame_variable_errors(thread, error_strings)
