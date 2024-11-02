"""
Test process attach.
"""



import os
import lldb
import shutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

exe_name = "ProcessAttach"  # Must match Makefile


class ProcessAttachTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Waiting to be attached...')

    @skipIfiOSSimulator
    def test_attach_to_process_by_id(self):
        """Test attach by process id"""
        self.build()
        exe = self.getBuildArtifact(exe_name)

        # Spawn a new process
        popen = self.spawnSubprocess(exe)

        self.runCmd("process attach -p " + str(popen.pid))

        target = self.dbg.GetSelectedTarget()

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)

    @skipIfiOSSimulator
    def test_attach_to_process_by_id_autocontinue(self):
        """Test attach by process id"""
        self.build()
        exe = self.getBuildArtifact(exe_name)

        # Spawn a new process
        popen = self.spawnSubprocess(exe)

        self.runCmd("process attach -c -p " + str(popen.pid))

        target = self.dbg.GetSelectedTarget()

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertTrue(process.GetState(), lldb.eStateRunning)

    @skipIfWindows # This is flakey on Windows AND when it fails, it hangs: llvm.org/pr48806
    def test_attach_to_process_from_different_dir_by_id(self):
        """Test attach by process id"""
        newdir = self.getBuildArtifact("newdir")
        try:
            os.mkdir(newdir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
        testdir = self.getBuildDir()
        exe = os.path.join(newdir, 'proc_attach')
        self.buildProgram('main.cpp', exe)
        self.addTearDownHook(lambda: shutil.rmtree(newdir))

        # Spawn a new process
        popen = self.spawnSubprocess(exe)

        os.chdir(newdir)
        self.addTearDownHook(lambda: os.chdir(testdir))
        self.runCmd("process attach -p " + str(popen.pid))

        target = self.dbg.GetSelectedTarget()

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)

    def test_attach_to_process_by_name(self):
        """Test attach by process name"""
        self.build()
        exe = self.getBuildArtifact(exe_name)

        # Spawn a new process
        popen = self.spawnSubprocess(exe)

        self.runCmd("process attach -n " + exe_name)

        target = self.dbg.GetSelectedTarget()

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)

    @skipIfWindows # This test is flaky on Windows
    @expectedFailureNetBSD
    def test_attach_to_process_by_id_correct_executable_offset(self):
        """
        Test that after attaching to a process the executable offset
        is determined correctly on FreeBSD.  This is a regression test
        for dyld plugin getting the correct executable path,
        and therefore being able to identify it in the module list.
        """

        self.build()
        exe = self.getBuildArtifact(exe_name)

        # In order to reproduce, we must spawn using a relative path
        popen = self.spawnSubprocess(os.path.relpath(exe))

        self.runCmd("process attach -p " + str(popen.pid))

        # Make sure we did not attach too early.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=False)
        self.runCmd("process continue")
        self.expect("v g_val", substrs=["12345"])

    def tearDown(self):
        # Destroy process before TestBase.tearDown()
        self.dbg.GetSelectedTarget().GetProcess().Destroy()

        # Call super's tearDown().
        TestBase.tearDown(self)
