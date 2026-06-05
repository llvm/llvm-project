"""
Test the pipe command (lldb.utils.pipe).
"""

import os
import tempfile

import lldb
from lldbsuite.test.lldbtest import *


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.runCmd("command script import lldb.utils.pipe")

    def test_pipe_to_grep(self):
        """Test piping an LLDB command through grep."""
        self.expect("pipe help help | grep Syntax", substrs=["Syntax"])

    def test_lldb_command_no_pipe(self):
        """Test running a plain LLDB command (passthrough)."""
        self.expect("pipe help help", substrs=["Syntax"])

    def test_pipe_chain(self):
        """Test piping through multiple shell commands."""
        self.expect("pipe help help | tr s-z S-Z | grep -i syntax", substrs=["SYnTaX"])

    def test_redirect_stdout(self):
        """Test redirecting LLDB command output to a file."""
        with tempfile.NamedTemporaryFile(mode="r", delete=False) as f:
            path = f.name
        try:
            self.runCmd(f"pipe help help > {path}")
            with open(path) as f:
                contents = f.read()
            self.assertIn("Syntax", contents)
        finally:
            os.unlink(path)

    def test_shell_command_no_pipe(self):
        """Test running a plain shell command (not an LLDB command)."""
        self.expect("pipe echo hello", substrs=["hello"])

    def test_shell_command_with_pipe(self):
        """Test running a shell command piped to another shell command."""
        self.expect("pipe echo hello world | tr a-z A-Z", substrs=["HELLO WORLD"])

    def test_quoted_pipe_not_split(self):
        """Test that a pipe character inside quotes is not treated as a split."""
        self.expect("pipe echo 'abc|def' | tr a-c A-C", substrs=["ABC|def"])

    def test_pipe_without_spaces(self):
        """Test that pipe works without spaces around the | operator."""
        self.expect("pipe help help|grep Syntax", substrs=["Syntax"])

    def test_error_no_command_before_pipe(self):
        """Test error when nothing precedes the pipe operator."""
        self.expect("pipe | grep foo", error=True)

    def test_failed_lldb_command(self):
        """Test that a non-lldb non-shell command reports an error."""
        self.expect("pipe not_a_real_lldb_command_xyz | wc -l", error=True)
