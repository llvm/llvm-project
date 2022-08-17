import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceDumpInfo(TraceIntelPTTestCaseBase):

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
            substrs=['''thread #1: tid = 3842849

  Trace technology: intel-pt

  Total number of trace items: 24

  Memory usage:
    Raw trace size: 4 KiB
    Total approximate memory usage (excluding raw trace): 0.21 KiB
    Average memory usage per item (excluding raw trace): 9.00 bytes

  Timing for this thread:
    Decoding instructions: ''', '''

  Events:
    Number of individual events: 3
      software disabled tracing: 2
      trace synchronization point: 1'''],
            patterns=["Decoding instructions: \d.\d\ds"])

    def testDumpRawTraceSizeJSON(self):
        self.expect("trace load -v " +
        os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
        substrs=["intel-pt"])

        self.expect("thread trace dump info --json ",
            substrs=['''{
  "traceTechnology": "intel-pt",
  "threadStats": {
    "tid": 3842849,
    "traceItemsCount": 24,
    "memoryUsage": {
      "totalInBytes": "216",
      "avgPerItemInBytes": 9
    },
    "timingInSeconds": {
      "Decoding instructions": 0''', '''
    },
    "events": {
      "totalCount": 3,
      "individualCounts": {
        "software disabled tracing": 2,
        "trace synchronization point": 1
      }
    }
  },
  "globalStats": {
    "timingInSeconds": {}
  }
}'''])
