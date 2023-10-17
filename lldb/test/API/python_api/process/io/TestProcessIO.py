"""Test Python APIs for process IO."""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ProcessIOTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setup_test(self):
        # Get the full path to our executable to be debugged.
        self.exe = self.getBuildArtifact("process_io")
        self.local_input_file = self.getBuildArtifact("input.txt")
        self.local_output_file = self.getBuildArtifact("output.txt")
        self.local_error_file = self.getBuildArtifact("error.txt")

        self.input_file = os.path.join(
            self.get_process_working_directory(), "input.txt"
        )
        self.output_file = os.path.join(
            self.get_process_working_directory(), "output.txt"
        )
        self.error_file = os.path.join(
            self.get_process_working_directory(), "error.txt"
        )
        self.lines = ["Line 1", "Line 2", "Line 3"]

    @skipIfWindows  # stdio manipulation unsupported on Windows
    @expectedFlakeyLinux(bugnumber="llvm.org/pr26437")
    @skipIfDarwinEmbedded  # I/O redirection like this is not supported on remote iOS devices yet <rdar://problem/54581135>
    def test_stdin_by_api(self):
        """Exercise SBProcess.PutSTDIN()."""
        self.setup_test()
        self.build()
        self.create_target()
        self.run_process(True)
        output = self.process.GetSTDOUT(1000)
        self.check_process_output(output, output)

    @skipIfWindows  # stdio manipulation unsupported on Windows
    @expectedFlakeyLinux(bugnumber="llvm.org/pr26437")
    def test_stdin_redirection(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDIN without specifying STDOUT or STDERR."""
        self.setup_test()
        self.build()
        self.create_target()
        self.redirect_stdin()
        self.run_process(False)
        output = self.process.GetSTDOUT(1000)
        self.check_process_output(output, output)

    @skipIfWindows  # stdio manipulation unsupported on Windows
    @expectedFlakeyLinux(bugnumber="llvm.org/pr26437")
    @skipIfDarwinEmbedded  # debugserver can't create/write files on the device
    def test_stdout_redirection(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDOUT without specifying STDIN or STDERR."""
        self.setup_test()
        self.build()
        self.create_target()
        self.redirect_stdout()
        self.run_process(True)
        output = self.read_output_file_and_delete()
        error = self.process.GetSTDOUT(1000)
        self.check_process_output(output, error)

    @skipIfWindows  # stdio manipulation unsupported on Windows
    @expectedFlakeyLinux(bugnumber="llvm.org/pr26437")
    @skipIfDarwinEmbedded  # debugserver can't create/write files on the device
    def test_stderr_redirection(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDERR without specifying STDIN or STDOUT."""
        self.setup_test()
        self.build()
        self.create_target()
        self.redirect_stderr()
        self.run_process(True)
        output = self.process.GetSTDOUT(1000)
        error = self.read_error_file_and_delete()
        self.check_process_output(output, error)

    @skipIfWindows  # stdio manipulation unsupported on Windows
    @expectedFlakeyLinux(bugnumber="llvm.org/pr26437")
    @skipIfDarwinEmbedded  # debugserver can't create/write files on the device
    def test_stdout_stderr_redirection(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDOUT and STDERR without redirecting STDIN."""
        self.setup_test()
        self.build()
        self.create_target()
        self.redirect_stdout()
        self.redirect_stderr()
        self.run_process(True)
        output = self.read_output_file_and_delete()
        error = self.read_error_file_and_delete()
        self.check_process_output(output, error)

    # target_file - path on local file system or remote file system if running remote
    # local_file - path on local system
    def read_file_and_delete(self, target_file, local_file):
        if lldb.remote_platform:
            self.runCmd(
                'platform get-file "{remote}" "{local}"'.format(
                    remote=target_file, local=local_file
                )
            )

        self.assertTrue(
            os.path.exists(local_file),
            'Make sure "{local}" file exists'.format(local=local_file),
        )
        f = open(local_file, "r")
        contents = f.read()
        f.close()

        # TODO: add 'platform delete-file' file command
        # if lldb.remote_platform:
        #    self.runCmd('platform delete-file "{remote}"'.format(remote=target_file))
        os.unlink(local_file)
        return contents

    def read_output_file_and_delete(self):
        return self.read_file_and_delete(self.output_file, self.local_output_file)

    def read_error_file_and_delete(self):
        return self.read_file_and_delete(self.error_file, self.local_error_file)

    def create_target(self):
        """Create the target and launch info that will be used by all tests"""
        self.target = self.dbg.CreateTarget(self.exe)
        self.launch_info = self.target.GetLaunchInfo()
        self.launch_info.SetWorkingDirectory(self.get_process_working_directory())

    def redirect_stdin(self):
        """Redirect STDIN (file descriptor 0) to use our input.txt file

        Make the input.txt file to use when redirecting STDIN, setup a cleanup action
        to delete the input.txt at the end of the test in case exceptions are thrown,
        and redirect STDIN in the launch info."""
        f = open(self.local_input_file, "w")
        for line in self.lines:
            f.write(line + "\n")
        f.close()

        if lldb.remote_platform:
            self.runCmd(
                'platform put-file "{local}" "{remote}"'.format(
                    local=self.local_input_file, remote=self.input_file
                )
            )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            os.unlink(self.local_input_file)
            # TODO: add 'platform delete-file' file command
            # if lldb.remote_platform:
            #    self.runCmd('platform delete-file "{remote}"'.format(remote=self.input_file))

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)
        self.launch_info.AddOpenFileAction(0, self.input_file, True, False)

    def redirect_stdout(self):
        """Redirect STDOUT (file descriptor 1) to use our output.txt file"""
        self.launch_info.AddOpenFileAction(1, self.output_file, False, True)

    def redirect_stderr(self):
        """Redirect STDERR (file descriptor 2) to use our error.txt file"""
        self.launch_info.AddOpenFileAction(2, self.error_file, False, True)

    def run_process(self, put_stdin):
        """Run the process to completion and optionally put lines to STDIN via the API if "put_stdin" is True"""
        # Set the breakpoints
        self.breakpoint = self.target.BreakpointCreateBySourceRegex(
            "Set breakpoint here", lldb.SBFileSpec("main.c")
        )
        self.assertTrue(self.breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        # This should launch the process and it should exit by the time we get back
        # because we have synchronous mode enabled
        self.process = self.target.Launch(self.launch_info, error)

        self.assertTrue(error.Success(), "Make sure process launched successfully")
        self.assertTrue(self.process, PROCESS_IS_VALID)

        if self.TraceOn():
            print("process launched.")

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, self.breakpoint
        )

        self.assertEqual(len(threads), 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        if self.TraceOn():
            print("process stopped at breakpoint, sending STDIN via LLDB API.")

        # Write data to stdin via the public API if we were asked to
        if put_stdin:
            for line in self.lines:
                self.process.PutSTDIN(line + "\n")

        # Let process continue so it will exit
        self.process.Continue()
        state = self.process.GetState()
        self.assertState(state, lldb.eStateExited, PROCESS_IS_VALID)

    def check_process_output(self, output, error):
        # Since we launched the process without specifying stdin/out/err,
        # a pseudo terminal is used for stdout/err, and we are satisfied
        # once "input line=>1" appears in stdout.
        # See also main.c.
        if self.TraceOn():
            print("output = '%s'" % output)
            print("error = '%s'" % error)

        for line in self.lines:
            check_line = "input line to stdout: %s" % (line)
            self.assertTrue(
                check_line in output, "verify stdout line shows up in STDOUT"
            )
        for line in self.lines:
            check_line = "input line to stderr: %s" % (line)
            self.assertTrue(
                check_line in error, "verify stderr line shows up in STDERR"
            )
