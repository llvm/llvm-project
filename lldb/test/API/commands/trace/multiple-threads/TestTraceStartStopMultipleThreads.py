import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceStartStopMultipleThreads(TraceIntelPTTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    @testSBAPIAndCommands
    def testStartMultipleLiveThreads(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("b 6")
        self.expect("b 11")

        self.expect("r")
        self.traceStartProcess()

        self.expect("continue")
        self.expect("thread trace dump instructions", substrs=['main.cpp:9'])

        # We'll see here the second thread
        self.expect("continue")
        self.expect("thread trace dump instructions", substrs=['main.cpp:4'])

        self.traceStopProcess()

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    @testSBAPIAndCommands
    def testStartMultipleLiveThreadsWithStops(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("b 6")
        self.expect("b 11")

        self.expect("r")
        self.traceStartProcess()

        # We'll see here the first thread
        self.expect("continue")

        # We are in thread 2
        self.expect("thread trace dump instructions", substrs=['main.cpp:9'])
        self.expect("thread trace dump instructions 2", substrs=['main.cpp:9'])

        # We stop tracing it
        self.expect("thread trace stop 2")

        # The trace is still in memory
        self.expect("thread trace dump instructions 2", substrs=['main.cpp:9'])

        # We'll stop at the next breakpoint, thread 2 will be still alive, but not traced. Thread 3 will be traced
        self.expect("continue")
        self.expect("thread trace dump instructions", substrs=['main.cpp:4'])
        self.expect("thread trace dump instructions 3", substrs=['main.cpp:4'])

        self.expect("thread trace dump instructions 2", substrs=['not traced'])

        self.traceStopProcess()

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    @testSBAPIAndCommands
    def testStartMultipleLiveThreadsWithStops(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("b 6")
        self.expect("b 11")

        self.expect("r")

        self.traceStartProcess()

        # We'll see here the first thread
        self.expect("continue")

        # We are in thread 2
        self.expect("thread trace dump instructions", substrs=['main.cpp:9'])
        self.expect("thread trace dump instructions 2", substrs=['main.cpp:9'])

        # We stop tracing all
        self.expect("thread trace stop all")

        # The trace is still in memory
        self.expect("thread trace dump instructions 2", substrs=['main.cpp:9'])

        # We'll stop at the next breakpoint in thread 3, thread 2 and 3 will be alive, but only 3 traced.
        self.expect("continue")
        self.expect("thread trace dump instructions", substrs=['main.cpp:4'])
        self.expect("thread trace dump instructions 3", substrs=['main.cpp:4'])
        self.expect("thread trace dump instructions 1", substrs=['not traced'])
        self.expect("thread trace dump instructions 2", substrs=['not traced'])

        self.traceStopProcess()

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testStartMultipleLiveThreadsWithThreadStartAll(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("b 6")
        self.expect("b 11")

        self.expect("r")

        self.expect("continue")
        # We are in thread 2
        self.expect("thread trace start all")
        # Now we have instructions in thread's 2 trace
        self.expect("n")

        self.expect("thread trace dump instructions 2", substrs=['main.cpp:11'])

        # We stop tracing all
        self.runCmd("thread trace stop all")

        # The trace is still in memory
        self.expect("thread trace dump instructions 2", substrs=['main.cpp:11'])

        # We'll stop at the next breakpoint in thread 3, and nothing should be traced
        self.expect("continue")
        self.expect("thread trace dump instructions 3", substrs=['not traced'])
        self.expect("thread trace dump instructions 1", substrs=['not traced'])
        self.expect("thread trace dump instructions 2", substrs=['not traced'])

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    @testSBAPIAndCommands
    def testStartMultipleLiveThreadsWithSmallTotalLimit(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("r")

        # trace the entire process with enough total size for 1 thread trace
        self.traceStartProcess(processBufferSizeLimit=5000)

        # we get the stop event when trace 2 appears and can't be traced
        self.expect("c", substrs=['Thread', "can't be traced"])
        # we get the stop event when trace 3 appears and can't be traced
        self.expect("c", substrs=['Thread', "can't be traced"])

        self.traceStopProcess()

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    @testSBAPIAndCommands
    def testStartPerCoreSession(self):
        self.skipIfPerCoreTracingIsNotSupported()

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("r")

        # We should fail if we hit the total buffer limit. Useful if the number
        # of cores is huge.
        self.traceStartProcess(error="True", processBufferSizeLimit=100,
            perCoreTracing=True,
            substrs=["The process can't be traced because the process trace size "
            "limit has been reached. Consider retracing with a higher limit."])

        self.traceStartProcess(perCoreTracing=True)
        self.traceStopProcess()

        self.traceStartProcess(perCoreTracing=True)
        # We can't support multiple per-core tracing sessions.
        self.traceStartProcess(error=True, perCoreTracing=True,
            substrs=["Process currently traced. Stop process tracing first"])

        # We can't support tracing per thread is per core is enabled.
        self.traceStartThread(
            error="True",
            substrs=["Process currently traced with per-core tracing. Stop process tracing first"])

        # We can't stop individual thread when per core is enabled.
        self.traceStopThread(error="True",
            substrs=["Can't stop tracing an individual thread when per-core process tracing is enabled"])

        # The GetState packet should return trace buffers per core and at least one traced thread
        self.expect("""process plugin packet send 'jLLDBTraceGetState:{"type":"intel-pt"}]'""",
            substrs=['''[{"kind":"traceBuffer","size":4096}],"coreId":''', '"tid":'])

        self.traceStopProcess()
