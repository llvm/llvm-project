import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceLoad(TraceIntelPTTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def testLoadMultiCoreTrace(self):
        src_dir = self.getSourceDir()
        trace_definition_file = os.path.join(src_dir, "intelpt-multi-core-trace", "trace.json")
        self.expect("trace load -v " + trace_definition_file, substrs=["intel-pt"])
        self.expect("thread trace dump instructions 2 -t",
          substrs=["19521: [tsc=0x008fb5211c143fd8] error: expected tracing enabled event",
                   "m.out`foo() + 65 at multi_thread.cpp:12:21",
                   "19520: [tsc=0x008fb5211bfbc69e] 0x0000000000400ba7    jg     0x400bb3"])
        self.expect("thread trace dump instructions 3 -t",
          substrs=["67910: [tsc=0x008fb5211bfdf270] 0x0000000000400bd7    addl   $0x1, -0x4(%rbp)",
                   "m.out`bar() + 26 at multi_thread.cpp:20:6"])

    def testLoadTrace(self):
        src_dir = self.getSourceDir()
        trace_definition_file = os.path.join(src_dir, "intelpt-trace", "trace.json")
        self.expect("trace load -v " + trace_definition_file, substrs=["intel-pt"])

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertEqual(process.GetProcessID(), 1234)

        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(process.GetThreadAtIndex(0).GetThreadID(), 3842849)

        self.assertEqual(target.GetNumModules(), 1)
        module = target.GetModuleAtIndex(0)
        path = module.GetFileSpec()
        self.assertEqual(path.fullpath, os.path.join(src_dir, "intelpt-trace", "a.out"))
        self.assertGreater(module.GetNumSections(), 0)
        self.assertEqual(module.GetSectionAtIndex(0).GetFileAddress(), 0x400000)

        self.assertEqual("6AA9A4E2-6F28-2F33-377D-59FECE874C71-5B41261A", module.GetUUIDString())

        # check that the Process and Thread objects were created correctly
        self.expect("thread info", substrs=["tid = 3842849"])
        self.expect("thread list", substrs=["Process 1234 stopped", "tid = 3842849"])
        self.expect("thread trace dump info", substrs=['''Trace technology: intel-pt

thread #1: tid = 3842849
  Total number of instructions: 21

  Memory usage:
    Raw trace size: 4 KiB
    Total approximate memory usage (excluding raw trace): 1.27 KiB
    Average memory usage per instruction (excluding raw trace): 61.76 bytes

  Timing for this thread:
    Decoding instructions: ''', '''

  Events:
    Number of instructions with events: 1
    Number of individual events: 1
      paused: 1

  Errors:
    Number of TSC decoding errors: 0'''])

    def testLoadInvalidTraces(self):
        src_dir = self.getSourceDir()
        # We test first an invalid type
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad.json"), error=True,
          substrs=['''error: expected object at traceSession.processes[0]

Context:
{
  "cpuInfo": { ... },
  "processes": [
    /* error: expected object */
    123
  ],
  "type": "intel-pt"
}

Schema:
{
  "type": "intel-pt",
  "cpuInfo": {
    // CPU information gotten from, for example, /proc/cpuinfo.

    "vendor": "GenuineIntel" | "unknown",
    "family": integer,
    "model": integer,
    "stepping": integer
  },'''])

        # Now we test a wrong cpu family field in the global session file
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad2.json"), error=True,
            substrs=['error: expected uint64_t at traceSession.cpuInfo.family', "Context", "Schema"])

        # Now we test a missing field in the intel-pt settings
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad4.json"), error=True,
            substrs=['''error: missing value at traceSession.cpuInfo.family

Context:
{
  "cpuInfo": /* error: missing value */ {
    "model": 79,
    "stepping": 1,
    "vendor": "GenuineIntel"
  },
  "processes": [],
  "type": "intel-pt"
}''', "Schema"])

        # Now we test an incorrect load address in the intel-pt settings
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad5.json"), error=True,
            substrs=['error: missing value at traceSession.processes[1].pid', "Schema"])

        # The following wrong schema will have a valid target and an invalid one. In the case of failure,
        # no targets should be created.
        self.assertEqual(self.dbg.GetNumTargets(), 0)
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad3.json"), error=True,
            substrs=['error: missing value at traceSession.processes[1].pid'])
        self.assertEqual(self.dbg.GetNumTargets(), 0)
