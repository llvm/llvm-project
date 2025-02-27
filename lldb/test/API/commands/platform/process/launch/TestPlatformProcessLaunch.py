"""
Test platform process launch.
"""

from textwrap import dedent
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import TestBase


class ProcessLaunchTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setup(self):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"))
        exe = lldbutil.append_to_process_working_directory(self, "a.out")
        outfile = lldbutil.append_to_process_working_directory(self, "stdio.log")
        return (exe, outfile)

    def test_process_launch_no_args(self):
        # When there are no extra arguments we just have 0, the program name.
        exe, outfile = self.setup()
        self.runCmd("platform process launch --stdout {} -s".format(outfile))
        self.runCmd("continue")

        stdio_log = lldbutil.read_file_on_target(self, outfile)
        self.assertEqual(
            dedent(
                """\
            Got 1 argument(s).
            [0]: {}
            """.format(
                    exe
                )
            ),
            stdio_log,
        )

    def test_process_launch_command_args(self):
        exe, outfile = self.setup()
        # Arguments given via the command override those in the settings.
        self.runCmd("settings set target.run-args D E")
        self.runCmd("platform process launch --stdout {} -s -- A B C".format(outfile))
        self.runCmd("continue")

        stdio_log = lldbutil.read_file_on_target(self, outfile)
        self.assertEqual(
            dedent(
                """\
            Got 4 argument(s).
            [0]: {}
            [1]: A
            [2]: B
            [3]: C
            """.format(
                    exe
                )
            ),
            stdio_log,
        )

    def test_process_launch_target_args(self):
        exe, outfile = self.setup()
        # When no arguments are passed via the command, use the setting.
        self.runCmd("settings set target.run-args D E")
        self.runCmd("platform process launch --stdout {} -s".format(outfile))
        self.runCmd("continue")

        stdio_log = lldbutil.read_file_on_target(self, outfile)
        self.assertEqual(
            dedent(
                """\
            Got 3 argument(s).
            [0]: {}
            [1]: D
            [2]: E
            """.format(
                    exe
                )
            ),
            stdio_log,
        )
