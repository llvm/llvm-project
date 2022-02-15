"""
Test saving a mini dump.
"""


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ProcessSaveCoreMinidumpTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_save_linux_mini_dump(self):
        """Test that we can save a Linux mini dump."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        core = self.getBuildArtifact("core.dmp")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory())
            self.assertEqual(process.GetState(), lldb.eStateStopped)

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
            self.runCmd("process save-core --plugin-name=minidump --style=stack " + core)
            self.assertTrue(os.path.isfile(core))
            self.assertSuccess(process.Kill())

            # To verify, we'll launch with the mini dump
            target = self.dbg.CreateTarget(None)
            process = target.LoadCore(core)

            # check if the core is in desired state
            self.assertTrue(process, PROCESS_IS_VALID)
            self.assertTrue(process.GetProcessInfo().IsValid())
            self.assertEqual(process.GetProcessInfo().GetProcessID(), expected_pid)
            self.assertTrue(target.GetTriple().find("linux") != -1)
            self.assertTrue(target.GetNumModules(), expected_number_of_modules)
            self.assertEqual(process.GetNumThreads(), expected_number_of_threads)

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
                self.assertTrue(thread_id in expected_threads)
        finally:
            # Clean up the mini dump file.
            self.assertTrue(self.dbg.DeleteTarget(target))
            if (os.path.isfile(core)):
                os.unlink(core)
