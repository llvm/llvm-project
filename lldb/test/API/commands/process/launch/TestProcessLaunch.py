"""
Test lldb process launch flags.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ProcessLaunchTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.runCmd("settings set auto-confirm true")

    def tearDown(self):
        self.runCmd("settings clear auto-confirm")
        TestBase.tearDown(self)

    @skipIfRemote
    def test_io(self):
        """Test that process launch I/O redirection flags work properly."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.expect("file " + exe, patterns=["Current executable set to .*a.out"])

        in_file = os.path.join(self.getSourceDir(), "input-file.txt")
        out_file = lldbutil.append_to_process_working_directory(self, "output-test.out")
        err_file = lldbutil.append_to_process_working_directory(self, "output-test.err")

        # Make sure the output files do not exist before launching the process
        try:
            os.remove(out_file)
        except OSError:
            pass

        try:
            os.remove(err_file)
        except OSError:
            pass

        launch_command = "process launch -i '{0}' -o '{1}' -e '{2}' -w '{3}'".format(
            in_file, out_file, err_file, self.get_process_working_directory()
        )

        if lldb.remote_platform:
            self.runCmd(
                'platform put-file "{local}" "{remote}"'.format(
                    local=in_file, remote=in_file
                )
            )

        self.expect(launch_command, patterns=["Process .* launched: .*a.out"])

        success = True
        err_msg = ""

        out = lldbutil.read_file_on_target(self, out_file)
        if out != "This should go to stdout.\n":
            success = False
            err_msg = (
                err_msg + "    ERROR: stdout file does not contain correct output.\n"
            )

        err = lldbutil.read_file_on_target(self, err_file)
        if err != "This should go to stderr.\n":
            success = False
            err_msg = (
                err_msg + "    ERROR: stderr file does not contain correct output.\n"
            )

        if not success:
            self.fail(err_msg)

    # rdar://problem/9056462
    # The process launch flag '-w' for setting the current working directory
    # not working?
    @skipIfRemote
    @expectedFailureAll(oslist=["freebsd", "linux"], bugnumber="llvm.org/pr20265")
    @expectedFailureNetBSD
    def test_set_working_dir_nonexisting(self):
        """Test that '-w dir' fails to set the working dir when running the inferior with a dir which doesn't exist."""
        d = {"CXX_SOURCES": "print_cwd.cpp"}
        self.build(dictionary=d)
        self.setTearDownCleanup(d)
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe)

        mywd = "my_working_dir"
        out_file_name = "my_working_dir_test.out"
        err_file_name = "my_working_dir_test.err"

        my_working_dir_path = self.getBuildArtifact(mywd)
        out_file_path = os.path.join(my_working_dir_path, out_file_name)
        err_file_path = os.path.join(my_working_dir_path, err_file_name)

        # Check that we get an error when we have a nonexisting path
        invalid_dir_path = mywd + "z"
        launch_command = "process launch -w %s -o %s -e %s" % (
            invalid_dir_path,
            out_file_path,
            err_file_path,
        )

        self.expect(
            launch_command,
            error=True,
            patterns=["error:.* No such file or directory: %s" % invalid_dir_path],
        )

    @skipIfRemote
    def test_set_working_dir_existing(self):
        """Test that '-w dir' sets the working dir when running the inferior."""
        d = {"CXX_SOURCES": "print_cwd.cpp"}
        self.build(dictionary=d)
        self.setTearDownCleanup(d)
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe)

        mywd = "my_working_dir"
        out_file_name = "my_working_dir_test.out"
        err_file_name = "my_working_dir_test.err"

        my_working_dir_path = self.getBuildArtifact(mywd)
        lldbutil.mkdir_p(my_working_dir_path)
        out_file_path = os.path.join(my_working_dir_path, out_file_name)
        err_file_path = os.path.join(my_working_dir_path, err_file_name)

        # Make sure the output files do not exist before launching the process
        try:
            os.remove(out_file_path)
            os.remove(err_file_path)
        except OSError:
            pass

        launch_command = "process launch -w %s -o %s -e %s" % (
            my_working_dir_path,
            out_file_path,
            err_file_path,
        )

        self.expect(launch_command, patterns=["Process .* launched: .*a.out"])

        success = True
        err_msg = ""

        # Check to see if the 'stdout' file was created
        try:
            out_f = open(out_file_path)
        except IOError:
            success = False
            err_msg = err_msg + "ERROR: stdout file was not created.\n"
        else:
            # Check to see if the 'stdout' file contains the right output
            line = out_f.readline()
            if self.TraceOn():
                print("line:", line)
            if not re.search(mywd, line):
                success = False
                err_msg = (
                    err_msg + "The current working directory was not set correctly.\n"
                )
                out_f.close()

        # Try to delete the 'stdout' and 'stderr' files
        try:
            os.remove(out_file_path)
            os.remove(err_file_path)
        except OSError:
            pass

        if not success:
            self.fail(err_msg)

    def test_environment_with_special_char(self):
        """Test that environment variables containing '*' and '}' are handled correctly by the inferior."""
        source = "print_env.cpp"
        d = {"CXX_SOURCES": source}
        self.build(dictionary=d)
        self.setTearDownCleanup(d)

        evil_var = "INIT*MIDDLE}TAIL"

        target = self.createTestTarget()
        main_source_spec = lldb.SBFileSpec(source)
        breakpoint = target.BreakpointCreateBySourceRegex(
            "// Set breakpoint here.", main_source_spec
        )

        process = target.LaunchSimple(
            None, ["EVIL=" + evil_var], self.get_process_working_directory()
        )
        self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(len(threads), 1)
        frame = threads[0].GetFrameAtIndex(0)
        sbvalue = frame.EvaluateExpression("evil")
        value = sbvalue.GetSummary().strip('"')

        self.assertEqual(value, evil_var)
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited, PROCESS_EXITED)
