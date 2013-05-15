"""
Test lldb logging.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class LogTestCase(TestBase):

    mydir = "logging"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym (self):
        self.buildDsym ()
        self.command_log_tests ("dsym")

    @dwarf_test
    def test_with_dwarf (self):
        self.buildDwarf ()
        self.command_log_tests ("dwarf")

    def command_log_tests (self, type):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe,
                    patterns = [ "Current executable set to .*a.out" ])

        log_file = os.path.join (os.getcwd(), "lldb-commands-log-%s-%s-%s.txt" % (type,
                                                                                  os.path.basename(self.getCompiler()),
                                                                                  self.getArchitecture()))

        if (os.path.exists (log_file)):
            os.remove (log_file)

        self.runCmd ("log enable -f '%s' lldb commands" % (log_file))
        
        self.runCmd ("command alias bp breakpoint")
                     
        self.runCmd ("bp set -n main")

        self.runCmd ("bp l")

        expected_log_lines = [
            "Processing command: command alias bp breakpoint\n",
            "HandleCommand, cmd_obj : 'command alias'\n",
            "HandleCommand, revised_command_line: 'command alias bp breakpoint'\n",
            "HandleCommand, wants_raw_input:'True'\n",
            "HandleCommand, command line after removing command name(s): 'bp breakpoint'\n",
            "HandleCommand, command succeeded\n",
            "Processing command: bp set -n main\n",
            "HandleCommand, cmd_obj : 'breakpoint set'\n",
            "HandleCommand, revised_command_line: 'breakpoint set -n main'\n",
            "HandleCommand, wants_raw_input:'False'\n",
            "HandleCommand, command line after removing command name(s): '-n main'\n",
            "HandleCommand, command succeeded\n",
            "Processing command: bp l\n",
            "HandleCommand, cmd_obj : 'breakpoint list'\n",
            "HandleCommand, revised_command_line: 'breakpoint l'\n",
            "HandleCommand, wants_raw_input:'False'\n",
            "HandleCommand, command line after removing command name(s): ''\n",
            "HandleCommand, command succeeded\n",
            ]

        self.assertTrue (os.path.isfile (log_file))

        idx = 0
        end = len (expected_log_lines)
        f = open (log_file)
        log_lines = f.readlines()
        f.close ()
        self.runCmd("log disable lldb")
        os.remove (log_file)

        err_msg = ""
        success = True

        if len (log_lines) != len (expected_log_lines):
            success = False
            err_msg = "Wrong number of lines in log file; expected: " + repr (len (expected_log_lines)) + " found: " + repr(len (log_lines))
        else:
            for line1, line2 in zip (log_lines, expected_log_lines):
                if line1 != line2:
                    success = False
                    err_msg = "Expected '" + line2 + "'; Found '" + line1 + "'"
                    break

        if not success:
            self.fail (err_msg)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

