import lldb
import json
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *


class TestTraceStartStopMultipleThreads(TraceIntelPTTestCaseBase):
    @skipIf(oslist=no_match(["linux"]), archs=no_match(["i386", "x86_64"]))
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
        self.expect("thread trace dump instructions", substrs=["main.cpp:9"])

        # We'll see here the second thread
        self.expect("continue")
        self.expect("thread trace dump instructions", substrs=["main.cpp:4"])

        self.traceStopProcess()

    @skipIf(oslist=no_match(["linux"]), archs=no_match(["i386", "x86_64"]))
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
        self.expect("thread trace dump instructions", substrs=["main.cpp:9"])
        self.expect("thread trace dump instructions 2", substrs=["main.cpp:9"])

        # We stop tracing it
        self.expect("thread trace stop 2")

        # The trace is still in memory
        self.expect("thread trace dump instructions 2", substrs=["main.cpp:9"])

        # We'll stop at the next breakpoint, thread 2 will be still alive, but not traced. Thread 3 will be traced
        self.expect("continue")
        self.expect("thread trace dump instructions", substrs=["main.cpp:4"])
        self.expect("thread trace dump instructions 3", substrs=["main.cpp:4"])

        self.expect("thread trace dump instructions 2", substrs=["not traced"])

        self.traceStopProcess()

    @skipIf(oslist=no_match(["linux"]), archs=no_match(["i386", "x86_64"]))
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
        self.expect("thread trace dump instructions", substrs=["main.cpp:9"])
        self.expect("thread trace dump instructions 2", substrs=["main.cpp:9"])

        # We stop tracing all
        self.expect("thread trace stop all")

        # The trace is still in memory
        self.expect("thread trace dump instructions 2", substrs=["main.cpp:9"])

        # We'll stop at the next breakpoint in thread 3, thread 2 and 3 will be alive, but only 3 traced.
        self.expect("continue")
        self.expect("thread trace dump instructions", substrs=["main.cpp:4"])
        self.expect("thread trace dump instructions 3", substrs=["main.cpp:4"])
        self.expect(
            "thread trace dump instructions 1", substrs=["not traced"], error=True
        )
        self.expect(
            "thread trace dump instructions 2", substrs=["not traced"], error=True
        )

        self.traceStopProcess()

    @skipIf(oslist=no_match(["linux"]), archs=no_match(["i386", "x86_64"]))
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

        self.expect("thread trace dump instructions 2", substrs=["main.cpp:11"])

        # We stop tracing all
        self.runCmd("thread trace stop all")

        # The trace is still in memory
        self.expect("thread trace dump instructions 2", substrs=["main.cpp:11"])

        # We'll stop at the next breakpoint in thread 3, and nothing should be traced
        self.expect("continue")
        self.expect(
            "thread trace dump instructions 3", substrs=["not traced"], error=True
        )
        self.expect(
            "thread trace dump instructions 1", substrs=["not traced"], error=True
        )
        self.expect(
            "thread trace dump instructions 2", substrs=["not traced"], error=True
        )

    @skipIf(oslist=no_match(["linux"]), archs=no_match(["i386", "x86_64"]))
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
        self.expect("c", substrs=["Thread", "can't be traced"])
        # we get the stop event when trace 3 appears and can't be traced
        self.expect("c", substrs=["Thread", "can't be traced"])

        self.traceStopProcess()

    @skipIf(oslist=no_match(["linux"]), archs=no_match(["i386", "x86_64"]))
    @testSBAPIAndCommands
    def testStartPerCpuSession(self):
        self.skipIfPerCpuTracingIsNotSupported()

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("r")

        # We should fail if we hit the total buffer limit. Useful if the number
        # of cpus is huge.
        self.traceStartProcess(
            error="True",
            processBufferSizeLimit=100,
            perCpuTracing=True,
            substrs=[
                "The process can't be traced because the process trace size "
                "limit has been reached. Consider retracing with a higher limit."
            ],
        )

        self.traceStartProcess(perCpuTracing=True)
        self.traceStopProcess()

        self.traceStartProcess(perCpuTracing=True)
        # We can't support multiple per-cpu tracing sessions.
        self.traceStartProcess(
            error=True,
            perCpuTracing=True,
            substrs=["Process currently traced. Stop process tracing first"],
        )

        # We can't support tracing per thread is per cpu is enabled.
        self.traceStartThread(
            error="True", substrs=["Thread with tid ", "is currently traced"]
        )

        # We can't stop individual thread when per cpu is enabled.
        self.traceStopThread(
            error="True",
            substrs=[
                "Can't stop tracing an individual thread when per-cpu process tracing is enabled"
            ],
        )

        # We move forward a little bit to collect some data
        self.expect("b 19")
        self.expect("c")

        # We will assert that the trace state will contain valid context switch and intel pt trace buffer entries.
        # Besides that, we need to get tsc-to-nanos conversion information.

        # We first parse the json response from the custom packet
        self.runCmd(
            """process plugin packet send 'jLLDBTraceGetState:{"type":"intel-pt"}]'"""
        )
        response_header = "response: "
        output = None
        for line in self.res.GetOutput().splitlines():
            if line.find(response_header) != -1:
                response = line[
                    line.find(response_header) + len(response_header) :
                ].strip()
                output = json.loads(response)

        self.assertTrue(output is not None)
        self.assertIn("cpus", output)
        self.assertIn("tscPerfZeroConversion", output)
        found_non_empty_context_switch = False

        for cpu in output["cpus"]:
            context_switch_size = None
            ipt_trace_size = None
            for binary_data in cpu["binaryData"]:
                if binary_data["kind"] == "iptTrace":
                    ipt_trace_size = binary_data["size"]
                elif binary_data["kind"] == "perfContextSwitchTrace":
                    context_switch_size = binary_data["size"]
            self.assertTrue(context_switch_size is not None)
            self.assertTrue(ipt_trace_size is not None)
            if context_switch_size > 0:
                found_non_empty_context_switch = True

        # We must have captured the context switch of when the target resumed
        self.assertTrue(found_non_empty_context_switch)

        self.expect("thread trace dump instructions")

        self.traceStopProcess()
