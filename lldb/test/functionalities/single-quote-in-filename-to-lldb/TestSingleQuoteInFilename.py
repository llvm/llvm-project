"""
Test the lldb command line takes a filename with single quote chars.
"""

import os
import unittest2
import lldb
import pexpect
from lldbtest import *

class SingleQuoteInCommandLineTestCase(TestBase):

    mydir = os.path.join("functionalities", "single-quote-in-filename-to-lldb")
    myexe = "path with '09/a.out"

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        try:
            os.remove("child_send.txt")
            os.remove("child_read.txt")
            os.remove(cls.myexe)
        except:
            pass

    def test_lldb_invocation_with_single_quote_in_filename(self):
        """Test that 'lldb my_file_name' works where my_file_name is a string with a single quote char in it."""
        self.buildDefault()
        system(["/bin/sh", "-c", "cp a.out \"%s\"" % self.myexe])

        # The default lldb prompt.
        prompt = "(lldb) "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s "%s"' % (self.lldbHere, self.lldbOption, self.myexe))
        child = self.child
        child.setecho(True)
        # Turn on logging for input/output to/from the child.
        with open('child_send.txt', 'w') as f_send:
            with open('child_read.txt', 'w') as f_read:
                child.logfile_send = f_send
                child.logfile_read = f_read

                child.expect_exact(prompt)

                child.send("help watchpoint")
                child.sendline('')
                child.expect_exact(prompt)

        # Now that the necessary logging is done, restore logfile to None to
        # stop further logging.
        child.logfile_send = None
        child.logfile_read = None
        
        with open('child_send.txt', 'r') as fs:
            if self.TraceOn():
                print "\n\nContents of child_send.txt:"
                print fs.read()
        with open('child_read.txt', 'r') as fr:
            from_child = fr.read()
            if self.TraceOn():
                print "\n\nContents of child_read.txt:"
                print from_child

            self.expect(from_child, exe=False,
                substrs = ["Current executable set to"])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
