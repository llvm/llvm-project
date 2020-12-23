"""
Test lldb-vscode setBreakpoints request
"""


import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import os
import shutil
import subprocess
import tempfile
import threading
import time


def spawn_and_wait(program, delay):
    if delay:
        time.sleep(delay)
    process = subprocess.Popen([program],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    process.wait()


class TestVSCode_attach(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def set_and_hit_breakpoint(self, continueToExit=True):
        source = 'main.c'
        breakpoint1_line = line_number(source, '// breakpoint 1')
        lines = [breakpoint1_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(len(breakpoint_ids), len(lines),
                         "expect correct number of breakpoints")
        self.continue_to_breakpoints(breakpoint_ids)
        if continueToExit:
            self.continue_to_exit()

    @skipIfWindows
    @skipIfNetBSD # Hangs on NetBSD as well
    @skipIfRemote
    def test_by_pid(self):
        '''
            Tests attaching to a process by process ID.
        '''
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")
        self.process = subprocess.Popen([program],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        self.attach(pid=self.process.pid)
        self.set_and_hit_breakpoint(continueToExit=True)

    @skipIfWindows
    @skipIfNetBSD # Hangs on NetBSD as well
    @skipIfRemote
    def test_by_name(self):
        '''
            Tests attaching to a process by process name.
        '''
        self.build_and_create_debug_adaptor()
        orig_program = self.getBuildArtifact("a.out")
        # Since we are going to attach by process name, we need a unique
        # process name that has minimal chance to match a process that is
        # already running. To do this we use tempfile.mktemp() to give us a
        # full path to a location where we can copy our executable. We then
        # run this copy to ensure we don't get the error "more that one
        # process matches 'a.out'".
        program = tempfile.mktemp()
        shutil.copyfile(orig_program, program)
        shutil.copymode(orig_program, program)

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(self,
            "pid_file_%d" % (int(time.time())))

        def cleanup():
            if os.path.exists(program):
                os.unlink(program)
            self.run_platform_command("rm %s" % (pid_file_path))
        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        popen = self.spawnSubprocess(program, [pid_file_path])

        pid = lldbutil.wait_for_file_on_target(self, pid_file_path)

        self.attach(program=program)
        self.set_and_hit_breakpoint(continueToExit=True)

    @skipUnlessDarwin
    @skipIfDarwin
    @skipIfNetBSD # Hangs on NetBSD as well
    def test_by_name_waitFor(self):
        '''
            Tests attaching to a process by process name and waiting for the
            next instance of a process to be launched, ingoring all current
            ones.
        '''
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")
        self.spawn_thread = threading.Thread(target=spawn_and_wait,
                                             args=(program, 1.0,))
        self.spawn_thread.start()
        self.attach(program=program, waitFor=True)
        self.set_and_hit_breakpoint(continueToExit=True)

    @skipIfWindows
    @skipIfDarwin
    @skipIfNetBSD # Hangs on NetBSD as well
    @skipIf(archs=["arm", "aarch64"]) # Example of a flaky run http://lab.llvm.org:8011/builders/lldb-aarch64-ubuntu/builds/5527/steps/test/logs/stdio
    def test_commands(self):
        '''
            Tests the "initCommands", "preRunCommands", "stopCommands",
            "exitCommands", "terminateCommands" and "attachCommands"
            that can be passed during attach.

            "initCommands" are a list of LLDB commands that get executed
            before the targt is created.
            "preRunCommands" are a list of LLDB commands that get executed
            after the target has been created and before the launch.
            "stopCommands" are a list of LLDB commands that get executed each
            time the program stops.
            "exitCommands" are a list of LLDB commands that get executed when
            the process exits
            "attachCommands" are a list of LLDB commands that get executed and
            must have a valid process in the selected target in LLDB after
            they are done executing. This allows custom commands to create any
            kind of debug session.
            "terminateCommands" are a list of LLDB commands that get executed when
            the debugger session terminates.
        '''
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")
        # Here we just create a target and launch the process as a way to test
        # if we are able to use attach commands to create any kind of a target
        # and use it for debugging
        attachCommands = [
            'target create -d "%s"' % (program),
            'process launch --stop-at-entry'
        ]
        initCommands = ['target list', 'platform list']
        preRunCommands = ['image list a.out', 'image dump sections a.out']
        stopCommands = ['frame variable', 'bt']
        exitCommands = ['expr 2+3', 'expr 3+4']
        terminateCommands = ['expr 4+2']
        self.attach(program=program,
                    attachCommands=attachCommands,
                    initCommands=initCommands,
                    preRunCommands=preRunCommands,
                    stopCommands=stopCommands,
                    exitCommands=exitCommands,
                    terminateCommands=terminateCommands)
        # Get output from the console. This should contain both the
        # "initCommands" and the "preRunCommands".
        output = self.get_console()
        # Verify all "initCommands" were found in console output
        self.verify_commands('initCommands', output, initCommands)
        # Verify all "preRunCommands" were found in console output
        self.verify_commands('preRunCommands', output, preRunCommands)

        functions = ['main']
        breakpoint_ids = self.set_function_breakpoints(functions)
        self.assertEquals(len(breakpoint_ids), len(functions),
                        "expect one breakpoint")
        self.continue_to_breakpoints(breakpoint_ids)
        output = self.get_console(timeout=1.0)
        self.verify_commands('stopCommands', output, stopCommands)

        # Continue after launch and hit the "pause()" call and stop the target.
        # Get output from the console. This should contain both the
        # "stopCommands" that were run after we stop.
        self.vscode.request_continue()
        time.sleep(0.5)
        self.vscode.request_pause()
        self.vscode.wait_for_stopped()
        output = self.get_console(timeout=1.0)
        self.verify_commands('stopCommands', output, stopCommands)

        # Continue until the program exits
        self.continue_to_exit()
        # Get output from the console. This should contain both the
        # "exitCommands" that were run after the second breakpoint was hit
        # and the "terminateCommands" due to the debugging session ending
        output = self.collect_console(duration=1.0)
        self.verify_commands('exitCommands', output, exitCommands)
        self.verify_commands('terminateCommands', output, terminateCommands)

    @skipIfWindows
    @skipIfDarwin
    @skipIfNetBSD # Hangs on NetBSD as well
    @skipIf(archs=["arm", "aarch64"]) # Example of a flaky run http://lab.llvm.org:8011/builders/lldb-aarch64-ubuntu/builds/5517/steps/test/logs/stdio
    def test_terminate_commands(self):
        '''
            Tests that the "terminateCommands", that can be passed during
            attach, are run when the debugger is disconnected.
        '''
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")
        # Here we just create a target and launch the process as a way to test
        # if we are able to use attach commands to create any kind of a target
        # and use it for debugging
        attachCommands = [
            'target create -d "%s"' % (program),
            'process launch'
        ]
        terminateCommands = ['expr 4+2']
        self.attach(program=program,
                    attachCommands=attachCommands,
                    terminateCommands=terminateCommands,
                    disconnectAutomatically=False)
        self.get_console()
        # Once it's disconnected the console should contain the
        # "terminateCommands"
        self.vscode.request_disconnect(terminateDebuggee=True)
        output = self.collect_console(duration=1.0)
        self.verify_commands('terminateCommands', output, terminateCommands)
