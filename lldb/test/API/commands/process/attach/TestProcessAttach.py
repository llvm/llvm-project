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
        self.line = line_number("main.cpp", "// Waiting to be attached...")

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

    @skipIfWindows  # This is flakey on Windows AND when it fails, it hangs: llvm.org/pr48806
    def test_attach_to_process_from_different_dir_by_id(self):
        """Test attach by process id"""
        newdir = self.getBuildArtifact("newdir")
        try:
            os.mkdir(newdir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
        testdir = self.getBuildDir()
        exe = os.path.join(newdir, "proc_attach")
        self.buildProgram("main.cpp", exe)
        self.addTearDownHook(lambda: shutil.rmtree(newdir))

        # Spawn a new process
        popen = self.spawnSubprocess(exe)

        os.chdir(newdir)
        sourcedir = self.getSourceDir()
        self.addTearDownHook(lambda: os.chdir(sourcedir))
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

    @skipIfWindows  # This test is flaky on Windows
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
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=False
        )
        self.runCmd("process continue")
        self.expect("v g_val", substrs=["12345"])

    def tearDown(self):
        # Destroy process before TestBase.tearDown()
        self.dbg.GetSelectedTarget().GetProcess().Destroy()

        # Call super's tearDown().
        TestBase.tearDown(self)

    # This test is flakey on Linux & Windows.  The failure mode is
    # that sometimes we miss the interrupt and never succeed in
    # getting out of the attach wait.
    @skipUnlessDarwin
    def test_run_then_attach_wait_interrupt(self):
        # Test that having run one process doesn't cause us to be unable
        # to interrupt a subsequent attach attempt.
        self.build()
        exe = self.getBuildArtifact(exe_name)

        target = lldbutil.run_to_breakpoint_make_target(self, exe_name, True)
        launch_info = target.GetLaunchInfo()
        launch_info.SetArguments(["q"], True)
        error = lldb.SBError()
        target.Launch(launch_info, error)
        self.assertSuccess(error, "Launched a process")
        self.assertState(target.process.state, lldb.eStateExited, "and it exited.") 
        
        # Okay now we've run a process, try to attach/wait to something
        # and make sure that we can interrupt that.
        
        options = lldb.SBCommandInterpreterRunOptions()
        options.SetPrintResults(True)
        options.SetEchoCommands(False)

        self.stdin_path = self.getBuildArtifact("stdin.txt")

        with open(self.stdin_path, "w") as input_handle:
            input_handle.write("process attach -w -n noone_would_use_this_name\nquit")

        # Python will close the file descriptor if all references
        # to the filehandle object lapse, so we need to keep one
        # around.
        self.filehandle = open(self.stdin_path, "r")
        self.dbg.SetInputFileHandle(self.filehandle, False)

        # No need to track the output
        self.stdout_path = self.getBuildArtifact("stdout.txt")
        self.out_filehandle = open(self.stdout_path, "w")
        self.dbg.SetOutputFileHandle(self.out_filehandle, False)
        self.dbg.SetErrorFileHandle(self.out_filehandle, False)

        n_errors, quit_req, crashed = self.dbg.RunCommandInterpreter(
            True, True, options, 0, False, False)
        
        while 1:
            time.sleep(1)
            if target.process.state == lldb.eStateAttaching:
                break

        self.dbg.DispatchInputInterrupt()
        self.dbg.DispatchInputInterrupt()

        # cycle waiting for the process state to change before trying
        # to read the command output.  I don't want to spin forever.
        counter = 0
        got_exit = False
        while counter < 20:
            if target.process.state == lldb.eStateExited:
                got_exit = True
                break
            counter += 1
            time.sleep(1)

        self.assertTrue(got_exit, "The process never switched to eStateExited")
        # Even if the state has flipped, we still need to wait for the
        # command to complete to see the result.  We don't have a way to
        # synchronize on "command completed" right now, but sleeping just
        # a bit should be enough, all that's left is passing this error
        # result to the command, and printing it to the debugger output.
        time.sleep(1)

        self.out_filehandle.flush()
        reader = open(self.stdout_path, "r")
        results = reader.readlines()
        found_result = False
        for line in results:
            if "Cancelled async attach" in line:
                found_result = True
                break
        if not found_result:
            print(f"Results: {results}")

        self.assertTrue(found_result, "Found async error in results")
