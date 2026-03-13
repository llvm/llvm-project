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
        self.line = line_number("a.c", "// Set file and line breakpoint inside a().")

    def test(self):
        """Break inside a() and b() defined within libfoo.a."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside a() by file and line first.
        lldbutil.run_break_set_by_file_and_line(
            self, "a.c", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Break at a(int) first.
        self.expect(
            "frame variable", VARIABLES_DISPLAYED_CORRECTLY, substrs=["(int) arg = 1"]
        )
        self.expect_var_path("__a_global", type="int", value="1")

        # Set breakpoint for b() next.
        lldbutil.run_break_set_by_symbol(
            self, "b", num_expected_locations=1, sym_exact=True
        )

        # Continue the program, we should break at b(int) next.
        self.runCmd("continue")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )
        self.expect(
            "frame variable", VARIABLES_DISPLAYED_CORRECTLY, substrs=["(int) arg = 2"]
        )
        self.expect_var_path("__b_global", type="int", value="2")

    def check_frame_variable_errors(self, thread, error_strings):
        command_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()
        result = interp.HandleCommand("frame variable", command_result)
        self.assertEqual(
            result, lldb.eReturnStatusFailed, "frame var succeeded unexpectedly"
        )
        command_error = command_result.GetError()

        frame = thread.GetFrameAtIndex(0)
        var_list = frame.GetVariables(True, True, False, True)
        self.assertEqual(var_list.GetSize(), 0)
        api_error = var_list.GetError().GetCString()

        for s in error_strings:
            self.assertIn(
                s,
                command_error,
                'Make sure "%s" exists in the command error "%s"' % (s, command_error),
            )
        for s in error_strings:
            self.assertIn(
                s,
                api_error,
                'Make sure "%s" exists in the API error "%s"' % (s, api_error),
            )

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
            self, "a", bkpt_module=exe
        )

        error_strings = [
            'debug map object file "',
            'libfoo.a(a.o)" containing debug info does not exist, debug info will not be loaded',
        ]
        self.check_frame_variable_errors(thread, error_strings)

    @skipIfRemote
    @skipIf(compiler="clang", compiler_version=["<", "12.0"])
    def test_archive_specifications(self):
        """
        Create archives and make sure the information we get when retrieving
        the modules specifications is correct.
        """
        self.build()
        libbar_path = self.getBuildArtifact("libbar.a")
        libfoo_path = self.getBuildArtifact("libfoo.a")
        libfoothin_path = self.getBuildArtifact("libfoo-thin.a")
        objfile_a = self.getBuildArtifact("a.o")
        objfile_b = self.getBuildArtifact("b.o")
        objfile_c = self.getBuildArtifact("c.o")
        size_a = os.path.getsize(objfile_a)
        size_b = os.path.getsize(objfile_b)
        size_c = os.path.getsize(objfile_c)

        # Test loading normal archives
        module_specs = lldb.SBModuleSpecList.GetModuleSpecifications(libfoo_path)
        num_specs = module_specs.GetSize()
        self.assertEqual(num_specs, 2)
        spec = module_specs.GetSpecAtIndex(0)
        self.assertEqual(spec.GetObjectName(), "a.o")
        self.assertEqual(spec.GetObjectSize(), size_a)
        spec = module_specs.GetSpecAtIndex(1)
        self.assertEqual(spec.GetObjectName(), "b.o")
        self.assertEqual(spec.GetObjectSize(), size_b)

        # Test loading thin archives
        module_specs = lldb.SBModuleSpecList.GetModuleSpecifications(libbar_path)
        num_specs = module_specs.GetSize()
        self.assertEqual(num_specs, 1)
        spec = module_specs.GetSpecAtIndex(0)
        self.assertEqual(spec.GetObjectName(), "c.o")
        self.assertEqual(spec.GetObjectSize(), size_c)

        module_specs = lldb.SBModuleSpecList.GetModuleSpecifications(libfoothin_path)
        num_specs = module_specs.GetSize()
        self.assertEqual(num_specs, 2)
        spec = module_specs.GetSpecAtIndex(0)
        self.assertEqual(spec.GetObjectName(), "a.o")
        self.assertEqual(spec.GetObjectSize(), size_a)
        spec = module_specs.GetSpecAtIndex(1)
        self.assertEqual(spec.GetObjectName(), "b.o")
        self.assertEqual(spec.GetObjectSize(), size_b, libfoothin_path)

    @skipIfRemote
    @skipUnlessDarwin
    def test_frame_var_errors_when_thin_archive_malformed(self):
        """
        Create thin archive libfoo.a and make it malformed to make sure
        we don't crash and report an appropriate error when resolving
        breakpoint using debug map.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")
        libfoo_path = self.getBuildArtifact("libfoo.a")
        libthin_path = self.getBuildArtifact("libfoo-thin.a")
        objfile_a = self.getBuildArtifact("a.o")
        objfile_b = self.getBuildArtifact("b.o")
        objfile_c = self.getBuildArtifact("c.o")
        # Replace the libfoo.a file with a thin archive containing the same
        # debug information (a.o, b.o). Then remove a.o from the file system
        # so we force an error when we set a breakpoint on a() function.
        # Since the a.o is missing, the debug info won't be loaded and we
        # should see an error when trying to break into a().
        os.remove(libfoo_path)
        shutil.copyfile(libthin_path, libfoo_path)
        os.remove(objfile_a)

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        # We won't be able to see source file
        self.expect(
            "b a",
            substrs=["Breakpoint 1: where = a.out`a, address ="],
        )
        # Break at a() should fail
        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(
            self, "a", bkpt_module=exe
        )
        error_strings = [
            '"a.o" object from the "',
            "libfoo.a\" archive: either the .o file doesn't exist in the archive or the modification time (0x",
            ") of the .o file doesn't match",
        ]
        self.check_frame_variable_errors(thread, error_strings)

        # Break at b() should succeed
        (target, process, thread, bkpt) = lldbutil.run_to_name_breakpoint(
            self, "b", bkpt_module=exe
        )
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )
        self.expect(
            "frame variable", VARIABLES_DISPLAYED_CORRECTLY, substrs=["(int) arg = 2"]
        )

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
            self, "a", bkpt_module=exe
        )

        error_strings = [
            '"a.o" object from the "',
            "libfoo.a\" archive: either the .o file doesn't exist in the archive or the modification time (0x",
            ") of the .o file doesn't match",
        ]
        self.check_frame_variable_errors(thread, error_strings)
