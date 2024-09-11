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
        self.assertSuccess(var.GetError(), "Didn't get counter variable")
        self.assertEqual(var.GetValueAsUnsigned(), value, "This was one-print")

    def copy_to_main(self, src, dst):
        # We are relying on the source file being newer than the .o file from
        # a previous build, so sleep a bit here to ensure that the touch is later.
        time.sleep(2)
        try:
            # Make sure dst is writeable before trying to write to it.
            if os.path.exists(dst):
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


    @skipIfWindows
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_TwoTargetsOneDebugger(self):
        self.do_test(False, True)

    @expectedFailureAll() # An external reference keeps modules alive
    @skipIfWindows
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_TwoTargetsOneDebuggerWithPin(self):
        self.do_test(False, True, True)

    @skipIfWindows
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_TwoTargetsTwoDebuggers(self):
        self.do_test(False, False)

    @expectedFailureAll() # An external reference keeps modules alive
    #@skipIfWindows
    #@skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_TwoTargetsTwoDebuggersWithPin(self):
        self.do_test(False, False, True)
        
    def do_test(self, one_target, one_debugger, use_pinning_module=False):
        # Make sure that if we have one target, and we run, then
        # change the binary and rerun, the binary (and any .o files
        # if using dwarf in .o file debugging) get removed from the
        # shared module cache.  They are no longer reachable.
        # If use_pinning_module is true, we make another SBModule that holds
        # a reference to the a.out, and will keep it alive.
        # At present, those tests fail.  But they show how easy it is to
        # strand a module so it doesn't go away.  We really should add a
        # Finalize to Modules so when lldb removes them from the
        # shared module cache, we delete the expensive parts of the Module
        # and mark it no longer Valid.  The we can test that that has
        # happened.
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

        self.pinning_module = None
        if use_pinning_module:
            self.pinning_module = target.FindModule(target.executable)
            self.assertTrue(self.pinning_module.IsValid(), "Valid pinning module")
            def cleanupPinningModule(self):
                if self.pinning_module:
                    self.pinning_module.Clear()
                    self.pinning_module = None
            self.addTearDownHook(cleanupPinningModule)
                
        # Make sure we ran the version we intended here:
        self.check_counter_var(thread, 1)
        process.Kill()

        # Now copy two-print.c over main.c, rebuild, and rerun:
        self.copy_to_main(two_print_path, main_c_path)

        self.build(dictionary={"C_SOURCES": main_c_path, "EXE": "a.out"})
        error = lldb.SBError()

        target2 = None
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
            if not one_target:
                new_debugger = lldb.SBDebugger().Create()
                new_debugger.SetAsync(False)
                self.old_debugger = self.dbg
                self.dbg = new_debugger

                def cleanupDebugger(self):
                    if self.old_debugger != None:
                        lldb.SBDebugger.Destroy(self.dbg)
                        self.dbg = self.old_debugger
                        self.old_debugger = None
                        # The testsuite teardown asserts if we haven't deleted all
                        # modules by the time we exit, so we have to clean this up.

                self.addTearDownHook(cleanupDebugger)
                (target2, process2, thread, bkpt) = lldbutil.run_to_source_breakpoint(
                    self, "return counter;", main_filespec
                )

        # In two-print.c counter will be 2:
        self.check_counter_var(thread, 2)

        # If we made two targets, destroy the first one, that should free up the
        # unreachable Modules:

        # The original debugger is the one that owns the first target:
        if not one_target:
            dbg = None
            if one_debugger:
                dbg = self.dbg
            else:
                dbg = self.old_debugger
            dbg.HandleCommand(f"target delete {dbg.GetIndexOfTarget(target)}")

        # For dSYM's there will be two lines of output, one for the a.out and one
        # for the dSYM, and no .o entries:
        num_a_dot_out_entries = 1
        num_main_dot_o_entries = 1
        if debug_style == "dsym":
            num_a_dot_out_entries += 1
            num_main_dot_o_entries -= 1

        # We shouldn't need the process anymore, that target holds onto the old
        # image list, so get rid of it now:
        if target2 and target2.process.IsValid():
            target2.process.Kill()

        error = self.check_image_list_result(num_a_dot_out_entries, 1)
        # Even if this fails, MemoryPressureDetected should fix this.
        if self.TraceOn():
            print("*** Calling MemoryPressureDetected")
        lldb.SBDebugger.MemoryPressureDetected()
        error_after_mpd = self.check_image_list_result(num_a_dot_out_entries,
                                                       num_main_dot_o_entries)
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
        interp.HandleCommand("image list -g -r -u -h -f -S", image_cmd_result)
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
