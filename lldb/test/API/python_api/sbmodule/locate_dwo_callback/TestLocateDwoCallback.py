"""
Test module locate dwo callback functionality
"""

import ctypes
import shutil
from lldbsuite.test.decorators import *
import lldb
from lldbsuite.test import lldbtest, lldbutil


class LocateDwoCallbackTestCase(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        lldbtest.TestBase.setUp(self)

        self.build()
        self.exe = self.getBuildArtifact("a.out")
        self.dwos = [
         self.getBuildArtifact("main.dwo"),
         self.getBuildArtifact("foo.dwo")
        ]

    def run_program(self):
        # Set a breakpoint at main
        target = self.dbg.CreateTarget(self.exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)
        lldbutil.run_break_set_by_symbol(self, "main")

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(self.process, lldbtest.PROCESS_IS_VALID)

    def check_symbolicated(self):
        thread = self.process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        self.assertEquals(len(frame.get_arguments()), 2)

    def check_not_symbolicated(self):
        thread = self.process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        self.assertNotEquals(len(frame.get_arguments()), 2)

    def moveDwos(self):
        """Move the dwos to a subdir in the build dir"""
        dwo_folder = os.path.join(self.getBuildDir(), "dwos")
        lldbutil.mkdir_p(dwo_folder)
        for dwo in self.dwos:
            shutil.move(dwo, dwo_folder)

    @skipIfWindows
    @skipIfDarwin
    def test_set_non_callable(self):
        with self.assertRaises(TypeError):
            lldb.SBModule.SetLocateDwoCallback("a")

    @skipIfWindows
    @skipIfDarwin
    def test_set_wrong_args(self):
        def test_args_less_than_5(a, b, c, d):
            pass
        with self.assertRaises(TypeError):
            lldb.SBModule.SetLocateDwoCallback(test_args_less_than_5)

    @skipIfWindows
    @skipIfDarwin
    def test_default_succeeds(self):
        lldb.SBModule.SetLocateDwoCallback(None)

        self.moveDwos()
        self.run_program()
        self.check_not_symbolicated()

    @skipIfWindows
    @skipIfDarwin
    def test_default_fails_when_dwos_moved(self):
        lldb.SBModule.SetLocateDwoCallback(None)

        self.moveDwos()
        self.run_program()
        self.check_not_symbolicated()

    @skipIfWindows
    @skipIfDarwin
    def test_callback_finds_dwos(self):
        lldb.SBModule.SetLocateDwoCallback(locate_dwo_callback)

        self.moveDwos()
        self.run_program()
        self.check_symbolicated()

        # We don't want to interfere with other tests, so we unset it here.
        lldb.SBModule.SetLocateDwoCallback(None)

    @skipIfWindows
    @skipIfDarwin
    def test_falls_back_when_dwo_fails(self):
        # Note that we *don't* move the dwo files here, so the callback
        # *shouldn't* actually find the files.
        lldb.SBModule.SetLocateDwoCallback(locate_dwo_callback)

        self.run_program()

        # We should properly fall back to the default.
        self.check_symbolicated()

        # We don't want to interfere with other tests, so we unset it here.
        lldb.SBModule.SetLocateDwoCallback(None)

def locate_dwo_callback(objfile, dwo_name, comp_dir, dwo_id, result_file_spec):
    import os
    dwo_dir_after_move = os.path.join(comp_dir, "dwos")

    if os.path.exists(os.path.join(dwo_dir_after_move, dwo_name)):
        result_file_spec.SetDirectory(dwo_dir_after_move)
        result_file_spec.SetFilename(dwo_name)

    return lldb.SBError()
