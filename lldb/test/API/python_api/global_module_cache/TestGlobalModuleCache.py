"""
Test the use of the global module cache in lldb
"""

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os
import shutil
from pathlib import Path
import time


class GlobalModuleCacheTestCase(TestBase):
    # NO_DEBUG_INFO_TESTCASE = True

    def check_counter_var(self, thread, value):
        frame = thread.frames[0]
        var = frame.FindVariable("counter")
        self.assertTrue(var.GetError().Success(), "Got counter variable")
        self.assertEqual(var.GetValueAsUnsigned(), value, "This was one-print")

    def copy_to_main(self, src, dst):
        # We are relying on the source file being newer than the .o file from
        # a previous build, so sleep a bit here to ensure that the touch is later.
        time.sleep(2)
        try:
            # Make sure dst is writeable before trying to write to it.
            subprocess.run(
                ["chmod", "777", dst],
                stdin=None,
                capture_output=False,
                encoding="utf-8",
            )
            shutil.copy(src, dst)
        except:
            self.fail(f"Could not copy {src} to {dst}")
        Path(dst).touch()

    # The rerun tests indicate rerunning on Windows doesn't really work, so
    # this one won't either.
    @skipIfWindows
    # On Arm and AArch64 Linux, this test attempts to pop a thread plan when
    # we only have the base plan remaining. Skip it until we can figure out
    # the bug this is exposing (https://github.com/llvm/llvm-project/issues/76057).
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_OneTargetOneDebugger(self):
        self.do_test(True, True)

    # This behaves as implemented but that behavior is not desirable.
    # This test tests for the desired behavior as an expected fail.
    @skipIfWindows
    @expectedFailureAll
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_TwoTargetsOneDebugger(self):
        self.do_test(False, True)

    @skipIfWindows
    @expectedFailureAll
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_OneTargetTwoDebuggers(self):
        self.do_test(True, False)

    def do_test(self, one_target, one_debugger):
        # Make sure that if we have one target, and we run, then
        # change the binary and rerun, the binary (and any .o files
        # if using dwarf in .o file debugging) get removed from the
        # shared module cache.  They are no longer reachable.
        debug_style = self.getDebugInfo()

        # Before we do anything, clear the global module cache so we don't
        # see objects from other runs:
        lldb.SBDebugger.MemoryPressureDetected()

        # Set up the paths for our two versions of main.c:
        main_c_path = os.path.join(self.getBuildDir(), "main.c")
        one_print_path = os.path.join(self.getSourceDir(), "one-print.c")
        two_print_path = os.path.join(self.getSourceDir(), "two-print.c")
        main_filespec = lldb.SBFileSpec(main_c_path)

        # First copy the one-print.c to main.c in the build folder and
        # build our a.out from there:
        self.copy_to_main(one_print_path, main_c_path)
        self.build(dictionary={"C_SOURCES": main_c_path, "EXE": "a.out"})

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "return counter;", main_filespec
        )

        # Make sure we ran the version we intended here:
        self.check_counter_var(thread, 1)
        process.Kill()

        # Now copy two-print.c over main.c, rebuild, and rerun:
        # os.unlink(target.GetExecutable().fullpath)
        self.copy_to_main(two_print_path, main_c_path)

        self.build(dictionary={"C_SOURCES": main_c_path, "EXE": "a.out"})
        error = lldb.SBError()
        if one_debugger:
            if one_target:
                (_, process, thread, _) = lldbutil.run_to_breakpoint_do_run(
                    self, target, bkpt
                )
            else:
                (target2, process2, thread, bkpt) = lldbutil.run_to_source_breakpoint(
                    self, "return counter;", main_filespec
                )
        else:
            if one_target:
                new_debugger = lldb.SBDebugger().Create()
                self.old_debugger = self.dbg
                self.dbg = new_debugger

                def cleanupDebugger(self):
                    lldb.SBDebugger.Destroy(self.dbg)
                    self.dbg = self.old_debugger
                    self.old_debugger = None

                self.addTearDownHook(cleanupDebugger)
                (target2, process2, thread, bkpt) = lldbutil.run_to_source_breakpoint(
                    self, "return counter;", main_filespec
                )

        # In two-print.c counter will be 2:
        self.check_counter_var(thread, 2)

        # If we made two targets, destroy the first one, that should free up the
        # unreachable Modules:
        if not one_target:
            target.Clear()

        num_a_dot_out_entries = 1
        # For dSYM's there will be two lines of output, one for the a.out and one
        # for the dSYM.
        if debug_style == "dsym":
            num_a_dot_out_entries += 1

        error = self.check_image_list_result(num_a_dot_out_entries, 1)
        # Even if this fails, MemoryPressureDetected should fix this.
        lldb.SBDebugger.MemoryPressureDetected()
        error_after_mpd = self.check_image_list_result(num_a_dot_out_entries, 1)
        fail_msg = ""
        if error != "":
            fail_msg = "Error before MPD: " + error

        if error_after_mpd != "":
            fail_msg = fail_msg + "\nError after MPD: " + error_after_mpd
        if fail_msg != "":
            self.fail(fail_msg)

    def check_image_list_result(self, num_a_dot_out, num_main_dot_o):
        # Check the global module list, there should only be one a.out, and if we are
        # doing dwarf in .o file, there should only be one .o file.  This returns
        # an error string on error - rather than asserting, so you can stage this
        # failing.
        image_cmd_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()
        interp.HandleCommand("image list -g", image_cmd_result)
        if self.TraceOn():
            print(f"Expected: a.out: {num_a_dot_out} main.o: {num_main_dot_o}")
            print(image_cmd_result)

        image_list_str = image_cmd_result.GetOutput()
        image_list = image_list_str.splitlines()
        found_a_dot_out = 0
        found_main_dot_o = 0

        for line in image_list:
            # FIXME: force this to be at the end of the string:
            if "a.out" in line:
                found_a_dot_out += 1
            if "main.o" in line:
                found_main_dot_o += 1

        if num_a_dot_out != found_a_dot_out:
            return f"Got {found_a_dot_out} number of a.out's, expected {num_a_dot_out}"

        if found_main_dot_o > 0 and num_main_dot_o != found_main_dot_o:
            return (
                f"Got {found_main_dot_o} number of main.o's, expected {num_main_dot_o}"
            )

        return ""
