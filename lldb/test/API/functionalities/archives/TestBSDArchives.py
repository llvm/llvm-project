"""Test breaking inside functions defined within a BSD archive file libfoo.a."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os
import time

class BSDArchivesTestCase(TestBase):

    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number in a(int) to break at.
        self.line = line_number(
            'a.c', '// Set file and line breakpoint inside a().')

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24527.  Makefile.rules doesn't know how to build static libs on Windows")
    @expectedFailureAll(remote=True)
    def test(self):
        """Break inside a() and b() defined within libfoo.a."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside a() by file and line first.
        lldbutil.run_break_set_by_file_and_line(
            self, "a.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Break at a(int) first.
        self.expect("frame variable", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['(int) arg = 1'])
        self.expect_var_path("__a_global", type="int", value="1")

        # Set breakpoint for b() next.
        lldbutil.run_break_set_by_symbol(
            self, "b", num_expected_locations=1, sym_exact=True)

        # Continue the program, we should break at b(int) next.
        self.runCmd("continue")
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])
        self.expect("frame variable", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['(int) arg = 2'])
        self.expect_var_path("__b_global", type="int", value="2")

        # Test loading thin archives
        archive_path = self.getBuildArtifact("libbar.a")
        module_specs = lldb.SBModuleSpecList.GetModuleSpecifications(archive_path)
        num_specs = module_specs.GetSize()
        self.assertEqual(num_specs, 1)
        self.assertEqual(module_specs.GetSpecAtIndex(0).GetObjectName(), "c.o")


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
            self.assertTrue(s in command_error, 'Make sure "%s" exists in the command error "%s"' % (s, command_error))
        for s in error_strings:
            self.assertTrue(s in api_error, 'Make sure "%s" exists in the API error "%s"' % (s, api_error))

    @skipIfRemote
    @skipUnlessDarwin
    def test_frame_var_errors_when_archive_missing(self):
        """
            Break inside a() and remove libfoo.a to make sure we can't load
            the debug information and report an appropriate error when doing
            'frame variable'.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        libfoo_path = self.getBuildArtifact("libfoo.a")
        # Delete the main.o file that contains the debug info so we force an
        # error when we run to main and try to get variables for the a()
        # function. Since the libfoo.a is missing, the debug info won't be
        # loaded and we should see an error when trying to read varibles.
        os.unlink(libfoo_path)

        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(
                self, 'a', bkpt_module=exe)

        error_strings = [
            'debug map object file "',
            'libfoo.a(a.o)" containing debug info does not exist, debug info will not be loaded'
        ]
        self.check_frame_variable_errors(thread, error_strings)

    @skipIfRemote
    @skipUnlessDarwin
    def test_frame_var_errors_when_mtime_mistmatch_for_object_in_archive(self):
        """
            Break inside a() and modify the modification time for "a.o" within
            libfoo.a to make sure we can't load the debug information and
            report an appropriate error when doing 'frame variable'.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        a_path = self.getBuildArtifact("a.o")

        # Change the modification time of the a.o object file after sleeping for
        # 2 seconds to ensure the modification time is different. The rebuild
        # only the "libfoo.a" target. This means the modification time of the
        # a.o within libfoo.a will not match the debug map's modification time
        # in a.out and will cause the debug information to not be loaded and we
        # should get an appropriate error when reading variables.
        time.sleep(2)
        os.utime(a_path, None)
        self.build(make_targets=["libfoo.a"])

        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(
                self, 'a', bkpt_module=exe)

        error_strings = [
            '"a.o" object from the "',
            'libfoo.a" archive: either the .o file doesn\'t exist in the archive or the modification time (0x',
            ') of the .o file doesn\'t match'
        ]
        self.check_frame_variable_errors(thread, error_strings)
