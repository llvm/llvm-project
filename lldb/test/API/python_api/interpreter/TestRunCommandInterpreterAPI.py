"""Test the RunCommandInterpreter API."""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class CommandRunInterpreterLegacyAPICase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    mydir = TestBase.compute_mydir(__file__)

    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def setUp(self):
        TestBase.setUp(self)

        self.stdin_path = self.getBuildArtifact("stdin.txt")

        with open(self.stdin_path, 'w') as input_handle:
            input_handle.write("nonexistingcommand\nquit")

        # Python will close the file descriptor if all references
        # to the filehandle object lapse, so we need to keep one
        # around.
        self.filehandle = open(self.stdin_path, 'r')
        self.dbg.SetInputFileHandle(self.filehandle, False)

        # No need to track the output
        self.devnull = open(os.devnull, 'w')
        self.dbg.SetOutputFileHandle(self.devnull, False)
        self.dbg.SetErrorFileHandle (self.devnull, False)

    @add_test_categories(['pyapi'])
    def test_run_session_with_error_and_quit_legacy(self):
        """Run non-existing and quit command returns appropriate values"""

        n_errors, quit_requested, has_crashed = self.dbg.RunCommandInterpreter(
                True, False, lldb.SBCommandInterpreterRunOptions(), 0, False,
                False)

        self.assertGreater(n_errors, 0)
        self.assertTrue(quit_requested)
        self.assertFalse(has_crashed)


class CommandRunInterpreterAPICase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    mydir = TestBase.compute_mydir(__file__)

    @skipIfReproducer # lldb::FileSP used in typemap cannot be instrumented.
    def setUp(self):
        TestBase.setUp(self)

        self.stdin_path = self.getBuildArtifact("stdin.txt")

        with open(self.stdin_path, 'w') as input_handle:
            input_handle.write("nonexistingcommand\nquit")

        self.dbg.SetInputFile(open(self.stdin_path, 'r'))

        # No need to track the output
        devnull = open(os.devnull, 'w')
        self.dbg.SetOutputFile(devnull)
        self.dbg.SetErrorFile(devnull)

    @add_test_categories(['pyapi'])
    def test_run_session_with_error_and_quit(self):
        """Run non-existing and quit command returns appropriate values"""

        n_errors, quit_requested, has_crashed = self.dbg.RunCommandInterpreter(
                True, False, lldb.SBCommandInterpreterRunOptions(), 0, False,
                False)

        self.assertGreater(n_errors, 0)
        self.assertTrue(quit_requested)
        self.assertFalse(has_crashed)
