"""
Test lldb logging.  This test just makes sure logging doesn't crash, and produces some output.
"""


import os
import json
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LogTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super(LogTestCase, self).setUp()
        self.log_file = self.getBuildArtifact("log-file.txt")

    def test_file_writing(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.expect("file " + exe, patterns=["Current executable set to .*a.out"])

        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # By default, Debugger::EnableLog() will set log options to
        # PREPEND_THREAD_NAME + OPTION_THREADSAFE. We don't want the
        # threadnames here, so we enable just threadsafe (-t).
        self.runCmd("log enable -f '%s' lldb commands" % (self.log_file))

        self.runCmd("command alias bp breakpoint")

        self.runCmd("bp set -n main")

        self.runCmd("bp l")

        self.runCmd("log disable lldb")

        self.assertTrue(os.path.isfile(self.log_file))

        with open(self.log_file, "r") as f:
            log_lines = f.read()
        os.remove(self.log_file)

        self.assertGreater(len(log_lines), 0, "Something was written to the log file.")

    # Check that lldb truncates its log files
    def test_log_truncate(self):
        # put something in our log file
        with open(self.log_file, "w") as f:
            for i in range(1, 1000):
                f.write("bacon\n")

        self.runCmd("log enable -f '%s' lldb commands" % self.log_file)
        self.runCmd("help log")
        self.runCmd("log disable lldb")

        self.assertTrue(os.path.isfile(self.log_file))
        with open(self.log_file, "r") as f:
            contents = f.read()

        # check that it got removed
        self.assertEqual(contents.find("bacon"), -1)

    # Check that lldb can append to a log file
    def test_log_append(self):
        # put something in our log file
        with open(self.log_file, "w") as f:
            f.write("bacon\n")

        self.runCmd("log enable -a -f '%s' lldb commands" % self.log_file)
        self.runCmd("help log")
        self.runCmd("log disable lldb")

        self.assertTrue(os.path.isfile(self.log_file))
        with open(self.log_file, "r") as f:
            contents = f.read()

        # check that it is still there
        self.assertEqual(contents.find("bacon"), 0)

    # Enable all log options and check that nothing crashes.
    @skipIfWindows
    def test_all_log_options(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        self.runCmd(
            "log enable -v -s -T -p -n -S -F -f '%s' lldb commands" % self.log_file
        )
        self.runCmd("help log")
        self.runCmd("log disable lldb")

        self.assertTrue(os.path.isfile(self.log_file))

    # Check that -j produces a valid JSONL log file where every line is a JSON
    # object and the requested metadata flags surface as JSON fields rather
    # than line prefixes.
    def test_json_output(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        self.runCmd("log enable -j -s -T -p -F -f '%s' lldb commands" % self.log_file)
        self.runCmd("help log")
        self.runCmd("log disable lldb")

        self.assertTrue(os.path.isfile(self.log_file))
        with open(self.log_file, "r") as f:
            lines = f.readlines()

        self.assertGreater(len(lines), 0, "log file should not be empty")
        for line in lines:
            # Every line must parse as JSON on its own.
            obj = json.loads(line)
            self.assertIsInstance(obj, dict)
            self.assertIn("message", obj)
            # The flags we passed must turn into JSON fields.
            self.assertIn("sequence", obj)
            self.assertIn("timestamp", obj)
            self.assertIn("pid", obj)
            self.assertIn("tid", obj)
            self.assertIn("file", obj)
            self.assertIn("function", obj)

    def test_log_invalid(self):
        self.expect(
            "log enable not_a_channel not_a_category",
            error=True,
            substrs=["Invalid log channel 'not_a_channel'"],
        )

        self.expect(
            "log enable lldb not_a_category",
            error=True,
            substrs=[
                "unrecognized log category 'not_a_category'",
                "Logging categories for 'lldb':",
            ],
        )

        self.expect(
            "log enable lldb not_a_category api not_a_category_either",
            error=True,
            substrs=[
                "unrecognized log categories 'not_a_category', 'not_a_category_either'",
                "Logging categories for 'lldb':",
            ],
        )
