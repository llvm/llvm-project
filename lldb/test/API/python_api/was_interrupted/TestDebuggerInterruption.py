"""
Test SBDebugger.InterruptRequested and SBCommandInterpreter.WasInterrupted.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
import threading
import os


class TestDebuggerInterruption(TestBase):
    """This test runs a command that starts up, rendevous with the test thread
       using threading barriers, then checks whether it has been interrupted.

    The command's first argument is either 'interp' or 'debugger', to test
    InterruptRequested and WasInterrupted respectively.

    The command has two modes, interrupt and check, the former is the one that
    waits for an interrupt.  Then latter just returns whether an interrupt was
    requested.  We use the latter to make sure we took down the flag correctly."""

    NO_DEBUG_INFO_TESTCASE = True

    class CommandRunner(threading.Thread):
        """This class is for running a command, and for making a thread to run the command on.
        It gets passed the test it is working on behalf of, and most of the important
        objects come from the test."""

        def __init__(self, test):
            super().__init__()
            self.test = test

        def rendevous(self):
            # We smuggle out barriers and event to the runner thread using thread local data:
            import interruptible

            interruptible.local_data = interruptible.BarrierContainer(
                self.test.before_interrupt_barrier,
                self.test.after_interrupt_barrier,
                self.test.event,
            )

    class DirectCommandRunner(CommandRunner):
        """ "This version runs a single command using HandleCommand."""

        def __init__(self, test, command):
            super().__init__(test)
            self.command = command

        def run(self):
            self.rendevous()
            result = self.test.dbg.GetCommandInterpreter().HandleCommand(
                self.command, self.test.result
            )
            if self.test.result_barrier:
                self.test.result_barrier.wait()

    class CommandInterpreterRunner(CommandRunner):
        """This version runs the CommandInterpreter and feeds the command to it."""

        def __init__(self, test):
            super().__init__(test)

        def run(self):
            self.rendevous()

            test = self.test

            # We will use files for debugger input and output:

            # First write down the command:
            with open(test.getBuildArtifact(test.in_filename), "w") as f:
                f.write(f"{test.command}\n")

            # Now set the debugger's stdout & stdin to our files, and run
            # the CommandInterpreter:
            with open(test.out_filename, "w") as outf, open(
                test.in_filename, "r"
            ) as inf:
                outsbf = lldb.SBFile(outf.fileno(), "w", False)
                orig_outf = test.dbg.GetOutputFile()
                error = test.dbg.SetOutputFile(outsbf)
                test.assertSuccess(error, "Could not set outfile")

                insbf = lldb.SBFile(inf.fileno(), "r", False)
                orig_inf = test.dbg.GetOutputFile()
                error = test.dbg.SetInputFile(insbf)
                test.assertSuccess(error, "Could not set infile")

                options = lldb.SBCommandInterpreterRunOptions()
                options.SetPrintResults(True)
                options.SetEchoCommands(False)

                test.dbg.RunCommandInterpreter(True, False, options, 0, False, False)
                test.dbg.GetOutputFile().Flush()

                error = test.dbg.SetOutputFile(orig_outf)
                test.assertSuccess(error, "Restored outfile")
                test.dbg.SetInputFile(orig_inf)
                test.assertSuccess(error, "Restored infile")

    def command_setup(self, args):
        """Insert our command, if needed.  Then set up event and barriers if needed.
        Then return the command to run."""

        self.interp = self.dbg.GetCommandInterpreter()
        self.command_name = "interruptible_command"
        self.cmd_result = lldb.SBCommandReturnObject()

        if not "check" in args:
            self.event = threading.Event()
            self.result_barrier = threading.Barrier(2, timeout=10)
            self.before_interrupt_barrier = threading.Barrier(2, timeout=10)
            self.after_interrupt_barrier = threading.Barrier(2, timeout=10)
        else:
            self.event = None
            self.result_barrier = None
            self.before_interrupt_barrier = None
            self.after_interrupt_barrier = None

        if not self.interp.UserCommandExists(self.command_name):
            # Make the command we're going to use - it spins calling WasInterrupted:
            cmd_filename = "interruptible"
            cmd_filename = os.path.join(self.getSourceDir(), "interruptible.py")
            self.runCmd(f"command script import {cmd_filename}")
            cmd_string = f"command script add {self.command_name} --class interruptible.WelcomeCommand"
            self.runCmd(cmd_string)

        if len(args) == 0:
            command = self.command_name
        else:
            command = self.command_name + " " + args
        return command

    def run_single_command(self, command):
        # Now start up a thread to run the command:
        self.result.Clear()
        self.runner = TestDebuggerInterruption.DirectCommandRunner(self, command)
        self.runner.start()

    def start_command_interp(self):
        self.runner = TestDebuggerInterruption.CommandInterpreterRunner(self)
        self.runner.start()

    def check_text(self, result_text, interrupted):
        if interrupted:
            self.assertIn(
                "Command was interrupted", result_text, "Got the interrupted message"
            )
        else:
            self.assertIn(
                "Command was not interrupted",
                result_text,
                "Got the not interrupted message",
            )

    def gather_output(self):
        # Now wait for the interrupt to interrupt the command:
        self.runner.join(10.0)
        finished = not self.runner.is_alive()
        # Don't leave the runner thread stranded if the interrupt didn't work.
        if not finished:
            self.event.set()
            self.runner.join(10.0)

        self.assertTrue(finished, "We did finish the command")

    def check_result(self, interrupted=True):
        self.gather_output()
        self.check_text(self.result.GetOutput(), interrupted)

    def check_result_output(self, interrupted=True):
        self.gather_output()
        buffer = ""
        # Okay, now open the file for reading, and read.
        with open(self.out_filename, "r") as f:
            buffer = f.read()

        self.assertNotEqual(len(buffer), 0, "No command data")
        self.check_text(buffer, interrupted)

    def debugger_interrupt_test(self, use_interrupt_requested):
        """Test that debugger interruption interrupts a command
        running directly through HandleCommand.
        If use_interrupt_requested is true, we'll check that API,
        otherwise we'll check WasInterrupted.  They should both do
        the same thing."""

        if use_interrupt_requested:
            command = self.command_setup("debugger")
        else:
            command = self.command_setup("interp")

        self.result = lldb.SBCommandReturnObject()
        self.run_single_command(command)

        # Okay now wait till the command has gotten started to issue the interrupt:
        self.before_interrupt_barrier.wait()
        # I'm going to do it twice here to test that it works as a counter:
        self.dbg.RequestInterrupt()
        self.dbg.RequestInterrupt()

        def cleanup():
            self.dbg.CancelInterruptRequest()

        self.addTearDownHook(cleanup)
        # Okay, now set both sides going:
        self.after_interrupt_barrier.wait()

        # Check that the command was indeed interrupted.  First rendevous
        # after the runner thread had a chance to execute the command:
        self.result_barrier.wait()
        self.assertTrue(self.result.Succeeded(), "Our command succeeded")
        result_output = self.result.GetOutput()
        self.check_result(True)

        # Do it again to make sure that the counter is counting:
        self.dbg.CancelInterruptRequest()
        command = self.command_setup("debugger")
        self.run_single_command(command)

        # This time we won't even get to run the command, since HandleCommand
        # checks for the interrupt state on entry, so we don't wait on the command
        # barriers.
        self.result_barrier.wait()

        # Again check that we were
        self.assertFalse(self.result.Succeeded(), "Our command was not allowed to run")
        error_output = self.result.GetError()
        self.assertIn(
            "... Interrupted", error_output, "Command was cut short by interrupt"
        )

        # Now take down the flag, and make sure that we aren't interrupted:
        self.dbg.CancelInterruptRequest()

        # Now make sure that we really did take down the flag:
        command = self.command_setup("debugger check")
        self.run_single_command(command)
        result_output = self.result.GetOutput()
        self.check_result(False)

    def test_debugger_interrupt_use_dbg(self):
        self.debugger_interrupt_test(True)

    def test_debugger_interrupt_use_interp(self):
        self.debugger_interrupt_test(False)

    def test_interp_doesnt_interrupt_debugger(self):
        """Test that interpreter interruption does not interrupt a command
        running directly through HandleCommand.
        If use_interrupt_requested is true, we'll check that API,
        otherwise we'll check WasInterrupted.  They should both do
        the same thing."""

        command = self.command_setup("debugger poll")

        self.result = lldb.SBCommandReturnObject()
        self.run_single_command(command)

        # Now raise the debugger interrupt flag.  It will also interrupt the command:
        self.before_interrupt_barrier.wait()
        self.dbg.GetCommandInterpreter().InterruptCommand()
        self.after_interrupt_barrier.wait()

        # Check that the command was indeed interrupted:
        self.result_barrier.wait()
        self.assertTrue(self.result.Succeeded(), "Our command succeeded")
        result_output = self.result.GetOutput()
        self.check_result(False)

    def interruptible_command_test(self, use_interrupt_requested):
        """Test that interpreter interruption interrupts a command
        running in the RunCommandInterpreter loop.
        If use_interrupt_requested is true, we'll check that API,
        otherwise we'll check WasInterrupted.  They should both do
        the same thing."""

        self.out_filename = self.getBuildArtifact("output")
        self.in_filename = self.getBuildArtifact("input")
        # We're going to overwrite the input file, but we
        # don't want data accumulating in the output file.

        if os.path.exists(self.out_filename):
            os.unlink(self.out_filename)

        # You should be able to use either check method interchangeably:
        if use_interrupt_requested:
            self.command = self.command_setup("debugger") + "\n"
        else:
            self.command = self.command_setup("interp") + "\n"

        self.start_command_interp()

        # Now give the interpreter a chance to run this command up
        # to the first barrier
        self.before_interrupt_barrier.wait()
        # Then issue the interrupt:
        sent_interrupt = self.dbg.GetCommandInterpreter().InterruptCommand()
        self.assertTrue(sent_interrupt, "Did send command interrupt.")
        # Now give the command a chance to finish:
        self.after_interrupt_barrier.wait()

        self.check_result_output(True)

        os.unlink(self.out_filename)

        # Now send the check command, and make sure the flag is now down.
        self.command = self.command_setup("interp check") + "\n"
        self.start_command_interp()

        self.check_result_output(False)

    def test_interruptible_command_check_dbg(self):
        self.interruptible_command_test(True)

    def test_interruptible_command_check_interp(self):
        self.interruptible_command_test(False)

    def test_debugger_doesnt_interrupt_command(self):
        """Test that debugger interruption doesn't interrupt a command
        running in the RunCommandInterpreter loop."""

        self.out_filename = self.getBuildArtifact("output")
        self.in_filename = self.getBuildArtifact("input")
        # We're going to overwrite the input file, but we
        # don't want data accumulating in the output file.

        if os.path.exists(self.out_filename):
            os.unlink(self.out_filename)

        self.command = self.command_setup("interp poll") + "\n"

        self.start_command_interp()

        self.before_interrupt_barrier.wait()
        self.dbg.RequestInterrupt()

        def cleanup():
            self.dbg.CancelInterruptRequest()

        self.addTearDownHook(cleanup)
        self.after_interrupt_barrier.wait()

        self.check_result_output(False)

        os.unlink(self.out_filename)
