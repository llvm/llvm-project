"""
Test lldb 'commands regex' command which allows the user to create a regular expression command.
"""

import os
import unittest2
import lldb
import pexpect
from lldbtest import *

class CommandRegexTestCase(TestBase):

    mydir = os.path.join("functionalities", "command_regex")

    def test_command_regex(self):
        """Test a simple scenario of 'command regexp' invocation and subsequent use."""
        prompt = "(lldb) "
        regex_prompt = "Enter regular expressions in the form 's/<regex>/<subst>/' and terminate with an empty line:\r\n"
        regex_prompt1 = "\r\n"

        child = pexpect.spawn('%s %s' % (self.lldbHere, self.lldbOption))
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout
        # So that the spawned lldb session gets shutdown durng teardown.
        self.child = child

        # Substitute 'Help!' for 'help' using the 'commands regex' mechanism.
        child.expect_exact(prompt)
        child.sendline("command regex 'Help__'")
        child.expect_exact(regex_prompt)
        child.sendline('s/^$/help/')
        child.expect_exact(regex_prompt1)
        child.sendline('')
        # Help!
        child.sendline('Help__')
        # If we see the familiar 'help' output, the test is done.
        child.expect('The following is a list of built-in, permanent debugger commands:')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
