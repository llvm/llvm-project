"""
Test quoting of arguments to lldb commands.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SettingsCommandTestCase(TestBase):
    output_file_name = "output.txt"

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        cls.RemoveTempFile(SettingsCommandTestCase.output_file_name)

    @no_debug_info_test
    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # No quotes.
        self.expect_args("a b c", "a\0b\0c\0")
        # Single quotes.
        self.expect_args("'a b c'", "a b c\0")
        # Double quotes.
        self.expect_args('"a b c"', "a b c\0")
        # Single quote escape.
        self.expect_args("'a b\\' c", "a b\\\0c\0")
        # Double quote escape.
        self.expect_args('"a b\\" c"', 'a b" c\0')
        self.expect_args('"a b\\\\" c', "a b\\\0c\0")
        # Single quote in double quotes.
        self.expect_args('"a\'b"', "a'b\0")
        # Double quotes in single quote.
        self.expect_args("'a\"b'", 'a"b\0')
        # Combined quotes.
        self.expect_args("\"a b\"c'd e'", "a bcd e\0")
        # Bare single/double quotes.
        self.expect_args("a\\'b", "a'b\0")
        self.expect_args('a\\"b', 'a"b\0')

    def expect_args(self, args_in, args_out):
        """Test argument parsing. Run the program with args_in. The program dumps its arguments
        to stdout. Compare the stdout with args_out."""

        filename = SettingsCommandTestCase.output_file_name
        outfile = self.getBuildArtifact(filename)

        if lldb.remote_platform:
            outfile_arg = os.path.join(
                lldb.remote_platform.GetWorkingDirectory(), filename
            )
        else:
            outfile_arg = outfile

        self.runCmd("process launch -- %s %s" % (outfile_arg, args_in))

        if lldb.remote_platform:
            src_file_spec = lldb.SBFileSpec(outfile_arg, False)
            dst_file_spec = lldb.SBFileSpec(outfile, True)
            lldb.remote_platform.Get(src_file_spec, dst_file_spec)

        with open(outfile, "r") as f:
            output = f.read()

        self.RemoveTempFile(outfile)
        self.assertEqual(output, args_out)
