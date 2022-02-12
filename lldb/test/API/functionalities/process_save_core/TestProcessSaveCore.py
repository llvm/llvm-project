"""
Test saving a core file (or mini dump).
"""


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ProcessSaveCoreTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote
    @skipUnlessWindows
    def test_cannot_save_core_unless_process_stopped(self):
        """Test that SaveCore fails if the process isn't stopped."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        core = self.getBuildArtifact("core.dmp")
        target = self.dbg.CreateTarget(exe)
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertNotEqual(process.GetState(), lldb.eStateStopped)
        error = process.SaveCore(core)
        self.assertTrue(error.Fail())

    @skipIfRemote
    @skipUnlessWindows
    def test_save_windows_mini_dump(self):
        """Test that we can save a Windows mini dump."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        core = self.getBuildArtifact("core.dmp")
        try:
            target = self.dbg.CreateTarget(exe)
            breakpoint = target.BreakpointCreateByName("bar")
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory())
            self.assertEqual(process.GetState(), lldb.eStateStopped)
            self.assertTrue(process.SaveCore(core))
            self.assertTrue(os.path.isfile(core))
            self.assertSuccess(process.Kill())

            # To verify, we'll launch with the mini dump, and ensure that we see
            # the executable in the module list.
            target = self.dbg.CreateTarget(None)
            process = target.LoadCore(core)
            files = [
                target.GetModuleAtIndex(i).GetFileSpec() for i in range(
                    0, target.GetNumModules())]
            paths = [
                os.path.join(
                    f.GetDirectory(),
                    f.GetFilename()) for f in files]
            self.assertIn(exe, paths)

        finally:
            # Clean up the mini dump file.
            self.assertTrue(self.dbg.DeleteTarget(target))
            if (os.path.isfile(core)):
                os.unlink(core)

    @skipUnlessPlatform(["freebsd", "netbsd"])
    def test_save_core_via_process_plugin(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        core = self.getBuildArtifact("a.out.core")
        try:
            target = self.dbg.CreateTarget(exe)
            breakpoint = target.BreakpointCreateByName("bar")
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory())
            self.assertEqual(process.GetState(), lldb.eStateStopped)
            self.assertTrue(process.SaveCore(core))
            self.assertTrue(os.path.isfile(core))
            self.assertSuccess(process.Kill())
            pid = process.GetProcessID()

            target = self.dbg.CreateTarget(None)
            process = target.LoadCore(core)
            self.assertTrue(process, PROCESS_IS_VALID)
            self.assertEqual(process.GetProcessID(), pid)

        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))
            try:
                os.unlink(core)
            except OSError:
                pass
