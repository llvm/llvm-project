"""
Test that lldb can target and debug an executable whose path is longer than the
Windows MAX_PATH limit (260 characters).
"""

import os
import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

MAX_PATH = 260


@skipUnlessWindows
class LongPathTargetTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _long_path(self, path):
        return "\\\\?\\" + os.path.abspath(path)

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

    def test_target_with_long_path(self):
        """CreateTarget, launch and break in an executable located past
        MAX_PATH, and verify the full path is preserved (not truncated)."""
        self.build()
        src_exe = self.getBuildArtifact("a.out")

        long_dir = self._make_long_dir()
        if long_dir is None:
            self.skipTest("OS cannot create paths longer than MAX_PATH")

        exe_basename = os.path.basename(src_exe)
        long_exe = os.path.join(long_dir, exe_basename)
        shutil.copyfile(src_exe, self._long_path(long_exe))

        long_exe_abs = os.path.abspath(long_exe)
        self.assertGreater(
            len(long_exe_abs),
            MAX_PATH,
            "the test executable path must exceed MAX_PATH to be meaningful",
        )

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
