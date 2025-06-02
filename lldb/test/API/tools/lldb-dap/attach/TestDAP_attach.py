"""
Test lldb-dap attach request
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import subprocess
import threading
import time


def spawn_and_wait(program, delay):
    if delay:
        time.sleep(delay)
    process = subprocess.Popen(
        [program], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    process.wait()


class TestDAP_attach(lldbdap_testcase.DAPTestCaseBase):
    def set_and_hit_breakpoint(self, continueToExit=True):
        source = "main.c"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        lines = [breakpoint1_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        # Test binary will sleep for 10s, offset the breakpoint timeout
        # accordingly.
        timeout_offset = 10
        self.continue_to_breakpoints(
            breakpoint_ids, timeout=timeout_offset + self.DEFAULT_TIMEOUT
        )
        if continueToExit:
            self.continue_to_exit()

    @skipIfNetBSD  # Hangs on NetBSD as well
    def test_by_pid(self):
        """
        Tests attaching to a process by process ID.
        """
        program = self.build_and_create_debug_adapter_for_attach()
        self.process = subprocess.Popen(
            [program],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.attach(pid=self.process.pid)
        self.set_and_hit_breakpoint(continueToExit=True)

    @skipIfNetBSD  # Hangs on NetBSD as well
    def test_by_name(self):
        """
        Tests attaching to a process by process name.
        """
        program = self.build_and_create_debug_adapter_for_attach()

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(
            self, "pid_file_%d" % (int(time.time()))
        )

        popen = self.spawnSubprocess(program, [pid_file_path])
        lldbutil.wait_for_file_on_target(self, pid_file_path)

        self.attach(program=program)
        self.set_and_hit_breakpoint(continueToExit=True)

    @skipUnlessDarwin
    @skipIfNetBSD  # Hangs on NetBSD as well
    def test_by_name_waitFor(self):
        """
        Tests attaching to a process by process name and waiting for the
        next instance of a process to be launched, ignoring all current
        ones.
        """
        program = self.build_and_create_debug_adapter_for_attach()
        self.spawn_thread = threading.Thread(
            target=spawn_and_wait,
            args=(
                program,
                1.0,
            ),
        )
        self.spawn_thread.start()
        self.attach(program=program, waitFor=True)
        self.set_and_hit_breakpoint(continueToExit=True)

    @skipIfNetBSD  # Hangs on NetBSD as well
    def test_commands(self):
        """
        Tests the "initCommands", "preRunCommands", "stopCommands",
        "exitCommands", "terminateCommands" and "attachCommands"
        that can be passed during attach.

        "initCommands" are a list of LLDB commands that get executed
        before the target is created.
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
        """
        program = self.build_and_create_debug_adapter_for_attach()

        # Here we just create a target and launch the process as a way to test
        # if we are able to use attach commands to create any kind of a target
        # and use it for debugging
        attachCommands = [
            'target create -d "%s"' % (program),
            "process launch --stop-at-entry",
        ]
        initCommands = ["target list", "platform list"]
        preRunCommands = ["image list a.out", "image dump sections a.out"]
        postRunCommands = ["help trace", "help process trace"]
        stopCommands = ["frame variable", "thread backtrace"]
        exitCommands = ["expr 2+3", "expr 3+4"]
        terminateCommands = ["expr 4+2"]
        self.attach(
            program=program,
            attachCommands=attachCommands,
            initCommands=initCommands,
            preRunCommands=preRunCommands,
            stopCommands=stopCommands,
            exitCommands=exitCommands,
            terminateCommands=terminateCommands,
            postRunCommands=postRunCommands,
        )
        # Get output from the console. This should contain both the
        # "initCommands" and the "preRunCommands".
        output = self.get_console()
        # Verify all "initCommands" were found in console output
        self.verify_commands("initCommands", output, initCommands)
        # Verify all "preRunCommands" were found in console output
        self.verify_commands("preRunCommands", output, preRunCommands)
        # Verify all "postRunCommands" were found in console output
        self.verify_commands("postRunCommands", output, postRunCommands)

        functions = ["main"]
        breakpoint_ids = self.set_function_breakpoints(functions)
        self.assertEqual(len(breakpoint_ids), len(functions), "expect one breakpoint")
        self.continue_to_breakpoints(breakpoint_ids)
        output = self.collect_console(timeout_secs=10, pattern=stopCommands[-1])
        self.verify_commands("stopCommands", output, stopCommands)

        # Continue after launch and hit the "pause()" call and stop the target.
        # Get output from the console. This should contain both the
        # "stopCommands" that were run after we stop.
        self.do_continue()
        time.sleep(0.5)
        self.dap_server.request_pause()
        self.dap_server.wait_for_stopped()
        output = self.collect_console(timeout_secs=10, pattern=stopCommands[-1])
        self.verify_commands("stopCommands", output, stopCommands)

        # Continue until the program exits
        self.continue_to_exit()
        # Get output from the console. This should contain both the
        # "exitCommands" that were run after the second breakpoint was hit
        # and the "terminateCommands" due to the debugging session ending
        output = self.collect_console(
            timeout_secs=10.0,
            pattern=terminateCommands[0],
        )
        self.verify_commands("exitCommands", output, exitCommands)
        self.verify_commands("terminateCommands", output, terminateCommands)

    def test_attach_command_process_failures(self):
        """
        Tests that a 'attachCommands' is expected to leave the debugger's
        selected target with a valid process.
        """
        program = self.build_and_create_debug_adapter_for_attach()
        attachCommands = ['script print("oops, forgot to attach to a process...")']
        resp = self.attach(
            program=program,
            attachCommands=attachCommands,
            expectFailure=True,
        )
        self.assertFalse(resp["success"])
        self.assertIn(
            "attachCommands failed to attach to a process",
            resp["body"]["error"]["format"],
        )

    @skipIfNetBSD  # Hangs on NetBSD as well
    def test_terminate_commands(self):
        """
        Tests that the "terminateCommands", that can be passed during
        attach, are run when the debugger is disconnected.
        """
        program = self.build_and_create_debug_adapter_for_attach()

        # Here we just create a target and launch the process as a way to test
        # if we are able to use attach commands to create any kind of a target
        # and use it for debugging
        attachCommands = [
            'target create -d "%s"' % (program),
            "process launch --stop-at-entry",
        ]
        terminateCommands = ["expr 4+2"]
        self.attach(
            program=program,
            attachCommands=attachCommands,
            terminateCommands=terminateCommands,
            disconnectAutomatically=False,
        )
        self.get_console()
        # Once it's disconnected the console should contain the
        # "terminateCommands"
        self.dap_server.request_disconnect(terminateDebuggee=True)
        output = self.collect_console(
            timeout_secs=1.0,
            pattern=terminateCommands[0],
        )
        self.verify_commands("terminateCommands", output, terminateCommands)
