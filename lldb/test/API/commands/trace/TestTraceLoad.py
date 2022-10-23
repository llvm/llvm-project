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
          substrs=["19532: [20456513.000 ns] (error) expected tracing enabled event",
                   "m.out`foo() + 65 at multi_thread.cpp:12:21",
                   "9524: [19691630.226 ns] 0x0000000000400ba7    jg     0x400bb3"])
        self.expect("thread trace dump instructions 3 -t",
          substrs=["61831: [19736134.073 ns] 0x0000000000400bd7    addl   $0x1, -0x4(%rbp)",
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
    "errors": {
      "totalCount": 0,
      "libiptErrors": {},
      "fatalErrors": 0,
      "otherErrors": 0
    },
    "continuousExecutions": 0,
    "PSBBlocks": 0
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
    "traceItemsCount": 19533,
    "memoryUsage": {
      "totalInBytes": "176065",
      "avgPerItemInBytes": 9.''', '''},
    "timingInSeconds": {
      "Decoding instructions": ''', '''
    },
    "events": {
      "totalCount": 11,
      "individualCounts": {
        "software disabled tracing": 1,
        "trace synchronization point": 1,
        "CPU core changed": 1,
        "HW clock tick": 8
      }
    },
    "errors": {
      "totalCount": 1,
      "libiptErrors": {},
      "fatalErrors": 0,
      "otherErrors": 1
    },
    "continuousExecutions": 1,
    "PSBBlocks": 1
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
          substrs=["19532: [20456513.000 ns] (error) expected tracing enabled event",
                   "m.out`foo() + 65 at multi_thread.cpp:12:21",
                   "19524: [19691630.226 ns] 0x0000000000400ba7    jg     0x400bb3"])
        self.expect("thread trace dump instructions 3 -t",
          substrs=["61831: [19736134.073 ns] 0x0000000000400bd7    addl   $0x1, -0x4(%rbp)",
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
          substrs=["19532: [20456513.000 ns] (error) expected tracing enabled event",
                   "m.out`foo() + 65 at multi_thread.cpp:12:21",
                   "19524: [19691630.226 ns] 0x0000000000400ba7    jg     0x400bb3"])
        self.expect("thread trace dump instructions 3 -t",
          substrs=["61831: [19736134.073 ns] 0x0000000000400bd7    addl   $0x1, -0x4(%rbp)",
                   "m.out`bar() + 26 at multi_thread.cpp:20:6"])

    @testSBAPIAndCommands
    def testLoadMultiCoreTraceWithMissingThreads(self):
        src_dir = self.getSourceDir()
        trace_description_file_path = os.path.join(src_dir, "intelpt-multi-core-trace", "trace_missing_threads.json")
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, substrs=["intel-pt"])
        self.expect("thread trace dump instructions 3 -t",
          substrs=["19532: [20456513.000 ns] (error) expected tracing enabled event",
                   "m.out`foo() + 65 at multi_thread.cpp:12:21",
                   "19524: [19691630.226 ns] 0x0000000000400ba7    jg     0x400bb3"])
        self.expect("thread trace dump instructions 2 -t",
          substrs=["61831: [19736134.073 ns] 0x0000000000400bd7    addl   $0x1, -0x4(%rbp)",
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
      trace synchronization point: 1'''])

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

    def testLoadTraceCursor(self):
        src_dir = self.getSourceDir()
        trace_description_file_path = os.path.join(src_dir, "intelpt-multi-core-trace", "trace.json")
        traceDescriptionFile = lldb.SBFileSpec(trace_description_file_path, True)

        error = lldb.SBError()
        trace = self.dbg.LoadTraceFromFile(error, traceDescriptionFile)
        self.assertSBError(error)

        target = self.dbg.GetSelectedTarget()
        process = target.process


        # 1. Test some expected items of thread 1's trace cursor.
        thread1 = process.threads[1]
        cursor = trace.CreateNewCursor(error, thread1)
        self.assertTrue(cursor)
        self.assertTrue(cursor.HasValue())
        cursor.Seek(0, lldb.eTraceCursorSeekTypeBeginning)
        cursor.SetForwards(True)

        self.assertTrue(cursor.IsEvent())
        self.assertEqual(cursor.GetEventTypeAsString(), "HW clock tick")
        self.assertEqual(cursor.GetCPU(), lldb.LLDB_INVALID_CPU_ID)

        cursor.Next()

        self.assertTrue(cursor.IsEvent())
        self.assertEqual(cursor.GetEventTypeAsString(), "CPU core changed")
        self.assertEqual(cursor.GetCPU(), 51)

        cursor.GoToId(19532)

        self.assertTrue(cursor.IsError())
        self.assertEqual(cursor.GetError(), "expected tracing enabled event")

        cursor.GoToId(19524)

        self.assertTrue(cursor.IsInstruction())
        self.assertEqual(cursor.GetLoadAddress(), 0x400BA7)



        # Helper function to check equality of the current item of two trace cursors.
        def assertCurrentTraceCursorItemEqual(lhs, rhs):
            self.assertTrue(lhs.HasValue() and rhs.HasValue())

            self.assertEqual(lhs.GetId(), rhs.GetId())
            self.assertEqual(lhs.GetItemKind(), rhs.GetItemKind())
            if lhs.IsError():
                self.assertEqual(lhs.GetError(), rhs.GetError())
            elif lhs.IsEvent():
                self.assertEqual(lhs.GetEventType(), rhs.GetEventType())
                self.assertEqual(lhs.GetEventTypeAsString(), rhs.GetEventTypeAsString())
            elif lhs.IsInstruction():
                self.assertEqual(lhs.GetLoadAddress(), rhs.GetLoadAddress())
            else:
                self.fail("Unknown trace item kind")

        for thread in process.threads:
            sequentialTraversalCursor = trace.CreateNewCursor(error, thread)
            self.assertSBError(error)
            # Skip threads with no trace items
            if not sequentialTraversalCursor.HasValue():
                continue

            # 2. Test "End" boundary of the trace by advancing past the trace's last item.
            sequentialTraversalCursor.Seek(0, lldb.eTraceCursorSeekTypeEnd)
            self.assertTrue(sequentialTraversalCursor.HasValue())
            sequentialTraversalCursor.SetForwards(True)
            sequentialTraversalCursor.Next()
            self.assertFalse(sequentialTraversalCursor.HasValue())



            # 3. Test sequential traversal using sequential access API (ie Next())
            # and random access API (ie GoToId()) simultaneously.
            randomAccessCursor = trace.CreateNewCursor(error, thread)
            self.assertSBError(error)
            # Reset the sequential cursor
            sequentialTraversalCursor.Seek(0, lldb.eTraceCursorSeekTypeBeginning)
            sequentialTraversalCursor.SetForwards(True)
            self.assertTrue(sequentialTraversalCursor.IsForwards())

            while sequentialTraversalCursor.HasValue():
                itemId = sequentialTraversalCursor.GetId()
                randomAccessCursor.GoToId(itemId)
                assertCurrentTraceCursorItemEqual(sequentialTraversalCursor, randomAccessCursor)
                sequentialTraversalCursor.Next()



            # 4. Test a random access with random access API (ie Seek()) and
            # sequential access API (ie consecutive calls to Next()).
            TEST_SEEK_ID = 3
            randomAccessCursor.GoToId(TEST_SEEK_ID )
            # Reset the sequential cursor
            sequentialTraversalCursor.Seek(0, lldb.eTraceCursorSeekTypeBeginning)
            sequentialTraversalCursor.SetForwards(True)
            for _ in range(TEST_SEEK_ID): sequentialTraversalCursor.Next()
            assertCurrentTraceCursorItemEqual(sequentialTraversalCursor, randomAccessCursor)

    @testSBAPIAndCommands
    def testLoadKernelTrace(self):
        # kernel section without loadAddress (using default loadAddress).
        src_dir = self.getSourceDir()
        trace_description_file_path = os.path.join(src_dir, "intelpt-kernel-trace", "trace.json")
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, substrs=["intel-pt"])

        self.expect("image list", substrs=["0xffffffff81000000", "modules/m.out"])

        self.expect("thread list", substrs=[
            "Process 1 stopped",
            "* thread #1: tid = 0x002d",
            "  thread #2: tid = 0x0033"])

        # kernel section with custom loadAddress.
        trace_description_file_path = os.path.join(src_dir, "intelpt-kernel-trace",
                "trace_with_loadAddress.json")
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, substrs=["intel-pt"])

        self.expect("image list", substrs=["0x400000", "modules/m.out"])

    @testSBAPIAndCommands
    def testLoadInvalidKernelTrace(self):
        src_dir = self.getSourceDir()

        # Test kernel section with non-empty processeses section.
        trace_description_file_path = os.path.join(src_dir, "intelpt-kernel-trace", "trace_kernel_with_process.json")
        expected_substrs = ['error: "processes" must be empty when "kernel" is provided when parsing traceBundle']
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, error=True, substrs=expected_substrs)

        # Test kernel section without cpus section.
        trace_description_file_path = os.path.join(src_dir, "intelpt-kernel-trace", "trace_kernel_wo_cpus.json")
        expected_substrs = ['error: "cpus" is required when "kernel" is provided when parsing traceBundle']
        self.traceLoad(traceDescriptionFilePath=trace_description_file_path, error=True, substrs=expected_substrs)
