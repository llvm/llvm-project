import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceDumpInfo(TraceIntelPTTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    def testErrorMessages(self):
        # We first check the output when there are no targets
        self.expect("thread trace dump info",
            substrs=["error: invalid target, create a target using the 'target create' command"],
            error=True)

        # We now check the output when there's a non-running target
        self.expect("target create " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))

        self.expect("thread trace dump info",
            substrs=["error: Command requires a current process."],
            error=True)

        # Now we check the output when there's a running target without a trace
        self.expect("b main")
        self.expect("run")

        self.expect("thread trace dump info",
            substrs=["error: Process is not being traced"],
            error=True)

    def testDumpRawTraceSize(self):
        self.expect("trace load -v " +
        os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
        substrs=["intel-pt"])

        self.expect("thread trace dump info",
            substrs=['''Trace technology: intel-pt

thread #1: tid = 3842849
  Raw trace size: 4 KiB
  Total number of instructions: 21
  Total approximate memory usage: 0.27 KiB
  Average memory usage per instruction: 13.00 bytes

  Number of TSC decoding errors: 0'''])
