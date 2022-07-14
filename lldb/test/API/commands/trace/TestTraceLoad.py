import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceLoad(TraceIntelPTTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    @testSBAPIAndCommands
    def testLoadMultiCoreTrace(self):
        src_dir = self.getSourceDir()
        trace_description_file_path = os.path.join(src_dir, "intelpt-multi-core-trace", "trace.json")
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, substrs=["intel-pt"])
        self.expect("thread trace dump instructions 2 -t",
          substrs=["19523: [tsc=40450075478109270] (error) expected tracing enabled event",
                   "m.out`foo() + 65 at multi_thread.cpp:12:21",
                   "19521: [tsc=40450075477657246] 0x0000000000400ba7    jg     0x400bb3"])
        self.expect("thread trace dump instructions 3 -t",
          substrs=["67912: [tsc=40450075477799536] 0x0000000000400bd7    addl   $0x1, -0x4(%rbp)",
                   "m.out`bar() + 26 at multi_thread.cpp:20:6"])

        self.expect("thread trace dump info --json",
          substrs=['''{
  "traceTechnology": "intel-pt",
  "threadStats": {
    "tid": 3497234,
    "traceItemsCount": 0,
    "memoryUsage": {
      "totalInBytes": "0",
      "avgPerItemInBytes": null
    },
    "timingInSeconds": {
      "Decoding instructions": ''', '''
    },
    "events": {
      "totalCount": 0,
      "individualCounts": {}
    },
    "continuousExecutions": 0,
    "PSBBlocks": 0,
    "errorItems": {
      "total": 0,
      "individualErrors": {}
    }
  },
  "globalStats": {
    "timingInSeconds": {
      "Context switch and Intel PT traces correlation": 0
    },
    "totalUnattributedPSBBlocks": 0,
    "totalCountinuosExecutions": 153,
    "totalPSBBlocks": 5,
    "totalContinuousExecutions": 153
  }
}'''])

        self.expect("thread trace dump info 2 --json",
          substrs=['''{
  "traceTechnology": "intel-pt",
  "threadStats": {
    "tid": 3497496,
    "traceItemsCount": 19524,
    "memoryUsage": {
      "totalInBytes": "175760",
      "avgPerItemInBytes": 9.''', '''},
    "timingInSeconds": {
      "Decoding instructions": ''', '''
    },
    "events": {
      "totalCount": 2,
      "individualCounts": {
        "software disabled tracing": 1,
        "CPU core changed": 1
      }
    },
    "continuousExecutions": 1,
    "PSBBlocks": 1,
    "errorItems": {
      "total": 0,
      "individualErrors": {}
    }
  },
  "globalStats": {
    "timingInSeconds": {
      "Context switch and Intel PT traces correlation": 0''', '''},
    "totalUnattributedPSBBlocks": 0,
    "totalCountinuosExecutions": 153,
    "totalPSBBlocks": 5,
    "totalContinuousExecutions": 153
  }
}'''])

    @testSBAPIAndCommands
    def testLoadCompactMultiCoreTrace(self):
        src_dir = self.getSourceDir()
        trace_description_file_path = os.path.join(src_dir, "intelpt-multi-core-trace", "trace.json")
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, substrs=["intel-pt"])

        self.expect("thread trace dump info 2", substrs=["Total number of continuous executions found: 153"])

        # we'll save the trace in compact format
        compact_trace_bundle_dir = os.path.join(self.getBuildDir(), "intelpt-multi-core-trace-compact")
        self.traceSave(compact_trace_bundle_dir, compact=True)

        # we'll delete the previous target and make sure it's trace object is deleted
        self.dbg.DeleteTarget(self.dbg.GetTargetAtIndex(0))
        self.expect("thread trace dump instructions 2 -t", substrs=["error: invalid target"], error=True)

        # we'll load the compact trace and make sure it works
        self.traceLoad(os.path.join(compact_trace_bundle_dir, "trace.json"), substrs=["intel-pt"])
        self.expect("thread trace dump instructions 2 -t",
          substrs=["19523: [tsc=40450075478109270] (error) expected tracing enabled event",
                   "m.out`foo() + 65 at multi_thread.cpp:12:21",
                   "19521: [tsc=40450075477657246] 0x0000000000400ba7    jg     0x400bb3"])
        self.expect("thread trace dump instructions 3 -t",
          substrs=["67912: [tsc=40450075477799536] 0x0000000000400bd7    addl   $0x1, -0x4(%rbp)",
                   "m.out`bar() + 26 at multi_thread.cpp:20:6"])

        # This reduced the number of continuous executions to look at
        self.expect("thread trace dump info 2", substrs=["Total number of continuous executions found: 3"])

        # We clean up for the next run of this test
        self.dbg.DeleteTarget(self.dbg.GetTargetAtIndex(0))

    @testSBAPIAndCommands
    def testLoadMultiCoreTraceWithStringNumbers(self):
        src_dir = self.getSourceDir()
        trace_description_file_path = os.path.join(src_dir, "intelpt-multi-core-trace", "trace_with_string_numbers.json")
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, substrs=["intel-pt"])
        self.expect("thread trace dump instructions 2 -t",
          substrs=["19523: [tsc=40450075478109270] (error) expected tracing enabled event",
                   "m.out`foo() + 65 at multi_thread.cpp:12:21",
                   "19521: [tsc=40450075477657246] 0x0000000000400ba7    jg     0x400bb3"])
        self.expect("thread trace dump instructions 3 -t",
          substrs=["67912: [tsc=40450075477799536] 0x0000000000400bd7    addl   $0x1, -0x4(%rbp)",
                   "m.out`bar() + 26 at multi_thread.cpp:20:6"])

    @testSBAPIAndCommands
    def testLoadMultiCoreTraceWithMissingThreads(self):
        src_dir = self.getSourceDir()
        trace_description_file_path = os.path.join(src_dir, "intelpt-multi-core-trace", "trace_missing_threads.json")
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, substrs=["intel-pt"])
        self.expect("thread trace dump instructions 3 -t",
          substrs=["19523: [tsc=40450075478109270] (error) expected tracing enabled event",
                   "m.out`foo() + 65 at multi_thread.cpp:12:21",
                   "19521: [tsc=40450075477657246] 0x0000000000400ba7    jg     0x400bb3"])
        self.expect("thread trace dump instructions 2 -t",
          substrs=["67912: [tsc=40450075477799536] 0x0000000000400bd7    addl   $0x1, -0x4(%rbp)",
                   "m.out`bar() + 26 at multi_thread.cpp:20:6"])

    @testSBAPIAndCommands
    def testLoadTrace(self):
        src_dir = self.getSourceDir()
        trace_description_file_path = os.path.join(src_dir, "intelpt-trace", "trace.json")
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, substrs=["intel-pt"])

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
        self.expect("thread trace dump info", substrs=['''thread #1: tid = 3842849

  Trace technology: intel-pt

  Total number of trace items: 23

  Memory usage:
    Raw trace size: 4 KiB
    Total approximate memory usage (excluding raw trace): 0.20 KiB
    Average memory usage per item (excluding raw trace): 9.00 bytes

  Timing for this thread:
    Decoding instructions: ''', '''

  Events:
    Number of individual events: 2
      software disabled tracing: 2

  Errors:
    Number of TSC decoding errors: 0'''])

    @testSBAPIAndCommands
    def testLoadInvalidTraces(self):
        src_dir = self.getSourceDir()

        # We test first an invalid type
        trace_description_file_path = os.path.join(src_dir, "intelpt-trace", "trace_bad.json")
        expected_substrs = ['''error: expected object at traceBundle.processes[0]

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
  },''']
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, error=True, substrs=expected_substrs)


        # Now we test a wrong cpu family field in the global bundle description file
        trace_description_file_path = os.path.join(src_dir, "intelpt-trace", "trace_bad2.json")
        expected_substrs = ['error: expected uint64_t at traceBundle.cpuInfo.family', "Context", "Schema"]
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, error=True, substrs=expected_substrs)


        # Now we test a missing field in the intel-pt settings
        trace_description_file_path = os.path.join(src_dir, "intelpt-trace", "trace_bad4.json")
        expected_substrs = ['''error: missing value at traceBundle.cpuInfo.family

Context:
{
  "cpuInfo": /* error: missing value */ {
    "model": 79,
    "stepping": 1,
    "vendor": "GenuineIntel"
  },
  "processes": [],
  "type": "intel-pt"
}''', "Schema"]
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, error=True, substrs=expected_substrs)


        # Now we test an incorrect load address in the intel-pt settings
        trace_description_file_path = os.path.join(src_dir, "intelpt-trace", "trace_bad5.json")
        expected_substrs = ['error: missing value at traceBundle.processes[1].pid', "Schema"]
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, error=True, substrs=expected_substrs)


        # The following wrong schema will have a valid target and an invalid one. In the case of failure,
        # no targets should be created.
        self.assertEqual(self.dbg.GetNumTargets(), 0)
        trace_description_file_path = os.path.join(src_dir, "intelpt-trace", "trace_bad3.json")
        expected_substrs = ['error: missing value at traceBundle.processes[1].pid']
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, error=True, substrs=expected_substrs)
        self.assertEqual(self.dbg.GetNumTargets(), 0)
