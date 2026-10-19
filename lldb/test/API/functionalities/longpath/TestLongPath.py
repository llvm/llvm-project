"""
Test that lldb can target and debug an executable whose path is longer than the
Windows MAX_PATH limit (260 characters).
"""

import os
import shutil
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

MAX_PATH = 260


@requireWindows
class LongPathTargetTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _long_path(self, path):
        return lldbutil.get_extended_windows_path(path)

    def _normalize(self, path):
        if path.startswith("\\\\?\\"):
            path = path[4:]
        return os.path.normcase(os.path.normpath(path))

    def _make_long_dir(self):
        """Create (and return) a directory whose absolute path comfortably
        exceeds MAX_PATH, or None if the OS refuses to create it."""
        components = [self.getBuildArtifact("deep")] + ["d" * 80] * 3
        target_dir = os.path.join(*components)
        try:
            os.makedirs(self._long_path(target_dir), exist_ok=True)
        except OSError:
            return None
        return target_dir

    def _find_module(self, target, long_exe):
        """Return the module in `target` whose path matches `long_exe`, or an
        invalid module if none does."""
        wanted = self._normalize(long_exe)
        for i in range(target.GetNumModules()):
            mod = target.GetModuleAtIndex(i)
            if self._normalize(mod.GetFileSpec().fullpath) == wanted:
                return mod
        return lldb.SBModule()

    def _copy_exe_to_long_dir(self):
        """Build the test program, copy it into a directory whose path exceeds
        MAX_PATH and return the path of the copy."""
        self.build()
        src_exe = self.getBuildArtifact("a.out")

        long_dir = self._make_long_dir()
        if long_dir is None:
            self.skipTest("OS cannot create paths longer than MAX_PATH")

        long_exe = os.path.join(long_dir, os.path.basename(src_exe))
        shutil.copyfile(src_exe, self._long_path(long_exe))

        self.assertGreater(
            len(os.path.abspath(long_exe)),
            MAX_PATH,
            "the test executable path must exceed MAX_PATH to be meaningful",
        )
        return long_exe

    def test_target_with_long_path(self):
        """CreateTarget, launch and break in an executable located past
        MAX_PATH, and verify the full path is preserved (not truncated)."""
        long_exe = self._copy_exe_to_long_dir()
        exe_basename = os.path.basename(long_exe)

        # Creating the target has to open and parse the file at the long path.
        target = self.dbg.CreateTarget(long_exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # The main executable module must report its full, untruncated path.
        module = self._find_module(target, long_exe)
        self.assertTrue(
            module.IsValid(),
            "the executable module should be found by its full long path",
        )
        self.assertGreater(
            len(module.GetFileSpec().fullpath), MAX_PATH, "module path truncated"
        )

        bp = target.BreakpointCreateByName("main", exe_basename)
        self.assertGreater(bp.GetNumLocations(), 0, "main breakpoint has a location")

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)
        self.assertState(
            process.GetState(), lldb.eStateStopped, "process stopped at main"
        )

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread, "stopped at the main breakpoint")
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "main")

        # After launch the loaded executable module still carries the full path.
        live_module = self._find_module(process.GetTarget(), long_exe)
        self.assertTrue(live_module.IsValid(), "loaded module found by long path")
        self.assertGreater(
            len(live_module.GetFileSpec().fullpath),
            MAX_PATH,
            "loaded module path truncated",
        )

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    def test_attach_with_long_path(self):
        """Attach by pid to a process whose executable is located past
        MAX_PATH, and verify lldb determines its architecture and stops it."""
        long_exe = self._copy_exe_to_long_dir()

        token = self.getBuildArtifact("token")
        # Pass the "\\?\" form so CreateProcessW accepts a path past MAX_PATH.
        popen = subprocess.Popen(
            [long_exe, token], executable=self._long_path(long_exe)
        )
        self.addTearDownHook(popen.kill)
        lldbutil.wait_for_file_on_target(self, token)

        # Start from an empty target so the architecture has to come from the
        # process.
        target = self.dbg.CreateTarget("")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(self.dbg.GetListener(), popen.pid, error)
        self.assertSuccess(error)
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)
        self.assertState(process.GetState(), lldb.eStateStopped)

        triple = target.GetTriple()
        self.assertTrue(triple, "target has a triple")
        self.assertNotEqual(triple.split("-")[0], "unknown", "target arch is known")
        # This reads the image file through Host::GetProcessInfo in lldb.
        self.assertTrue(
            process.GetProcessInfo().GetTriple(), "process info has a triple"
        )

        self.assertGreater(process.GetNumThreads(), 0, "process has threads")
        frame = process.GetSelectedThread().GetFrameAtIndex(0)
        self.assertTrue(frame.GetModule().IsValid(), "frame 0 is in a module")

        exe_module = target.FindModule(lldb.SBFileSpec(os.path.basename(long_exe)))
        self.assertTrue(exe_module.IsValid(), "the executable module is loaded")
