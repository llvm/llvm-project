"""Test that lldb command 'process signal SIGUSR1' to send a signal to the inferior works."""

import os, time, signal
import unittest2
import lldb
from lldbtest import *
import lldbutil

class SendSignalTestCase(TestBase):

    mydir = os.path.join("functionalities", "signal")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test that lldb command 'process signal SIGUSR1' sends a signal to the inferior process."""
        self.buildDsym()
        self.send_signal()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test that lldb command 'process signal SIGUSR1' sends a signal to the inferior process."""
        self.buildDwarf()
        self.send_signal()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', 'Put breakpoint here')

    def send_signal(self):
        """Test that lldb command 'process signal SIGUSR1' sends a signal to the inferior process."""

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main() function and immediately send a signal to the inferior after resuming.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.runCmd("thread backtrace")

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        self.runCmd("process status")
        output = self.res.GetOutput()
        pid = re.match("Process (.*) stopped", output).group(1)

        # After resuming the process, send it a SIGUSR1 signal.

        # It is necessary at this point to make command interpreter interaction
        # be asynchronous, because we want to resume the process and to send it
        # a signal.
        self.dbg.SetAsync(True)
        self.runCmd("process continue")
        # Insert a delay of 1 second before doing the signaling stuffs.
        time.sleep(1)

        self.runCmd("process handle -n False -p True -s True SIGUSR1")
        #os.kill(int(pid), signal.SIGUSR1)
        self.runCmd("process signal SIGUSR1")

        # Insert a delay of 1 second before checking the process status.
        time.sleep(1)
        # Make the interaction mode be synchronous again.
        self.dbg.SetAsync(False)
        self.expect("process status", STOPPED_DUE_TO_SIGNAL,
            startstr = "Process %s stopped" % pid,
            substrs = ['stop reason = signal SIGUSR1'])
        self.expect("thread backtrace", STOPPED_DUE_TO_SIGNAL,
            substrs = ['stop reason = signal SIGUSR1'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
