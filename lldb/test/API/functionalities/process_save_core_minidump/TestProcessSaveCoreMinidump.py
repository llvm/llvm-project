"""
Test saving a mini dump.
"""


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ProcessSaveCoreMinidumpTestCase(TestBase):
    def verify_core_file(
        self, core_path, expected_pid, expected_modules, expected_threads
    ):
        # To verify, we'll launch with the mini dump
        target = self.dbg.CreateTarget(None)
        process = target.LoadCore(core_path)

        # check if the core is in desired state
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertTrue(process.GetProcessInfo().IsValid())
        self.assertEqual(process.GetProcessInfo().GetProcessID(), expected_pid)
        self.assertNotEqual(target.GetTriple().find("linux"), -1)
        self.assertTrue(target.GetNumModules(), len(expected_modules))
        self.assertEqual(process.GetNumThreads(), len(expected_threads))

        for module, expected in zip(target.modules, expected_modules):
            self.assertTrue(module.IsValid())
            module_file_name = module.GetFileSpec().GetFilename()
            expected_file_name = expected.GetFileSpec().GetFilename()
            # skip kernel virtual dynamic shared objects
            if "vdso" in expected_file_name:
                continue
            self.assertEqual(module_file_name, expected_file_name)
            self.assertEqual(module.GetUUIDString(), expected.GetUUIDString())

        for thread_idx in range(process.GetNumThreads()):
            thread = process.GetThreadAtIndex(thread_idx)
            self.assertTrue(thread.IsValid())
            thread_id = thread.GetThreadID()
            self.assertIn(thread_id, expected_threads)
        self.dbg.DeleteTarget(target)

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_save_linux_mini_dump(self):
        """Test that we can save a Linux mini dump."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        core_stack = self.getBuildArtifact("core.stack.dmp")
        core_dirty = self.getBuildArtifact("core.dirty.dmp")
        core_full = self.getBuildArtifact("core.full.dmp")
        core_sb_stack = self.getBuildArtifact("core_sb.stack.dmp")
        core_sb_dirty = self.getBuildArtifact("core_sb.dirty.dmp")
        core_sb_full = self.getBuildArtifact("core_sb.full.dmp")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)

            # get neccessary data for the verification phase
            process_info = process.GetProcessInfo()
            expected_pid = process_info.GetProcessID() if process_info.IsValid() else -1
            expected_number_of_modules = target.GetNumModules()
            expected_modules = target.modules
            expected_number_of_threads = process.GetNumThreads()
            expected_threads = []

            for thread_idx in range(process.GetNumThreads()):
                thread = process.GetThreadAtIndex(thread_idx)
                thread_id = thread.GetThreadID()
                expected_threads.append(thread_id)

            # save core and, kill process and verify corefile existence
            base_command = "process save-core --plugin-name=minidump "
            self.runCmd(base_command + " --style=stack '%s'" % (core_stack))
            self.assertTrue(os.path.isfile(core_stack))
            self.verify_core_file(
                core_stack, expected_pid, expected_modules, expected_threads
            )

            self.runCmd(base_command + " --style=modified-memory '%s'" % (core_dirty))
            self.assertTrue(os.path.isfile(core_dirty))
            self.verify_core_file(
                core_dirty, expected_pid, expected_modules, expected_threads
            )

            self.runCmd(base_command + " --style=full '%s'" % (core_full))
            self.assertTrue(os.path.isfile(core_full))
            self.verify_core_file(
                core_full, expected_pid, expected_modules, expected_threads
            )

            # validate saving via SBProcess
            error = process.SaveCore(core_sb_stack, "minidump", lldb.eSaveCoreStackOnly)
            self.assertTrue(error.Success())
            self.assertTrue(os.path.isfile(core_sb_stack))
            self.verify_core_file(
                core_sb_stack, expected_pid, expected_modules, expected_threads
            )

            error = process.SaveCore(core_sb_dirty, "minidump", lldb.eSaveCoreDirtyOnly)
            self.assertTrue(error.Success())
            self.assertTrue(os.path.isfile(core_sb_dirty))
            self.verify_core_file(
                core_sb_dirty, expected_pid, expected_modules, expected_threads
            )

            # Minidump can now save full core files, but they will be huge and
            # they might cause this test to timeout.
            error = process.SaveCore(core_sb_full, "minidump", lldb.eSaveCoreFull)
            self.assertTrue(error.Success())
            self.assertTrue(os.path.isfile(core_sb_full))
            self.verify_core_file(
                core_sb_full, expected_pid, expected_modules, expected_threads
            )

            self.assertSuccess(process.Kill())
        finally:
            # Clean up the mini dump file.
            self.assertTrue(self.dbg.DeleteTarget(target))
            if os.path.isfile(core_stack):
                os.unlink(core_stack)
            if os.path.isfile(core_dirty):
                os.unlink(core_dirty)
            if os.path.isfile(core_full):
                os.unlink(core_full)
            if os.path.isfile(core_sb_stack):
                os.unlink(core_sb_stack)
            if os.path.isfile(core_sb_dirty):
                os.unlink(core_sb_dirty)
            if os.path.isfile(core_sb_full):
                os.unlink(core_sb_full)
