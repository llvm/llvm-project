"""Test Python APIs for process IO."""

import os, sys, time
import unittest2
import lldb
from lldbtest import *

class ProcessIOTestCase(TestBase):

    mydir = os.path.join("python_api", "process", "io")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_put_stdin_with_dsym(self):
        """Exercise SBProcess.PutSTDIN()."""
        self.buildDsym()
        self.put_stdin()

    @python_api_test
    @dwarf_test
    def test_put_stdin_with_dwarf(self):
        """Exercise SBProcess.PutSTDIN()."""
        self.buildDwarf()
        self.put_stdin()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Get the full path to our executable to be debugged.
        self.exe = os.path.join(os.getcwd(), "process_io")

    def put_stdin(self):
        """Launch a process and use SBProcess.PutSTDIN() to write data to it."""

        target = self.dbg.CreateTarget(self.exe)

        # Perform synchronous interaction with the debugger.
        self.setAsync(True)

        process = target.LaunchSimple(None, None, os.getcwd())
        if self.TraceOn():
            print "process launched."

        self.assertTrue(process, PROCESS_IS_VALID)

        process.PutSTDIN("Line 1 Entered.\n")
        process.PutSTDIN("Line 2 Entered.\n")
        process.PutSTDIN("Line 3 Entered.\n")

        for i in range(5):
            output = process.GetSTDOUT(500)
            error = process.GetSTDERR(500)
            if self.TraceOn():
                print "output->|%s|" % output
            # Since we launched the process without specifying stdin/out/err,
            # a pseudo terminal is used for stdout/err, and we are satisfied
            # once "input line=>1" appears in stdout.
            # See also main.c.
            if "input line=>1" in output:
                return
            time.sleep(5)

        self.fail("Expected output form launched process did not appear?")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
