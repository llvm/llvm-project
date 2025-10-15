"""
Test lldb-dap attach commands
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import time


class TestDAP_attachCommands(lldbdap_testcase.DAPTestCaseBase):
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
        output = self.collect_console(timeout=10, pattern=stopCommands[-1])
        self.verify_commands("stopCommands", output, stopCommands)

        # Continue after launch and hit the "pause()" call and stop the target.
        # Get output from the console. This should contain both the
        # "stopCommands" that were run after we stop.
        self.do_continue()
        time.sleep(0.5)
        self.dap_server.request_pause()
        self.dap_server.wait_for_stopped()
        output = self.collect_console(timeout=10, pattern=stopCommands[-1])
        self.verify_commands("stopCommands", output, stopCommands)

        # Continue until the program exits
        self.continue_to_exit()
        # Get output from the console. This should contain both the
        # "exitCommands" that were run after the second breakpoint was hit
        # and the "terminateCommands" due to the debugging session ending
        output = self.collect_console(
            timeout=10.0,
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
            timeout=1.0,
            pattern=terminateCommands[0],
        )
        self.verify_commands("terminateCommands", output, terminateCommands)
