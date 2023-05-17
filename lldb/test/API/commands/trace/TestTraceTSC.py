import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceTimestampCounters(TraceIntelPTTestCaseBase):

    @testSBAPIAndCommands
    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testTscPerThread(self):
        self.expect("file " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")

        self.traceStartThread(enableTsc=True)

        self.expect("n")
        self.expect("thread trace dump instructions -t -c 1",
            patterns=[": \[\d+.\d+ ns\] 0x0000000000400511    movl"])

    @testSBAPIAndCommands
    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testMultipleTscsPerThread(self):
        self.expect("file " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")

        self.traceStartThread(enableTsc=True)

        # After each stop there'll be a new TSC
        self.expect("si")
        self.expect("si")
        self.expect("si")

        # We'll get the most recent instructions, with at least 3 different TSCs
        self.runCmd("thread trace dump instructions -t --raw --forward")
        id_to_timestamp = {}
        for line in self.res.GetOutput().splitlines():
            m = re.search("    (.+): \[(.+)\ ns].*", line)
            if m:
                id_to_timestamp[int(m.group(1))] = m.group(2)
        self.assertEqual(len(id_to_timestamp), 3)

        # We check that the values are right when dumping a specific id
        for id, timestamp in id_to_timestamp.items():
            self.expect(f"thread trace dump instructions -t --id {id} -c 1",
                substrs=[f"{id}: [{timestamp} ns]"])

    @testSBAPIAndCommands
    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testTscPerProcess(self):
        self.expect("file " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")

        self.traceStartProcess(enableTsc=True)

        self.expect("n")
        self.expect("thread trace dump instructions -t -c 1",
            patterns=[": \[\d+.\d+ ns\] 0x0000000000400511    movl"])

        self.expect("thread trace dump instructions -t -c 1 --pretty-json",
            patterns=['''"timestamp_ns": "\d+.\d+"'''])

    @testSBAPIAndCommands
    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testDumpingAfterTracingWithoutTsc(self):
        self.expect("file " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")

        self.traceStartThread(enableTsc=False)

        self.expect("n")
        self.expect("thread trace dump instructions -t -c 1",
            patterns=[": \[unavailable\] 0x0000000000400511    movl"])

        self.expect("thread trace dump instructions -t -c 1 --json",
            substrs=['''"timestamp_ns":null'''])

    @testSBAPIAndCommands
    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testPSBPeriod(self):
        def isPSBSupported():
            caps_file = "/sys/bus/event_source/devices/intel_pt/caps/psb_cyc"
            if not os.path.exists(caps_file):
                return False
            with open(caps_file, "r") as f:
                val = int(f.readline())
                if val != 1:
                    return False
            return True

        def getValidPSBValues():
            values_file = "/sys/bus/event_source/devices/intel_pt/caps/psb_periods"
            values = []
            with open(values_file, "r") as f:
                mask = int(f.readline(), 16)
                for i in range(0, 32):
                    if (1 << i) & mask:
                        values.append(i)
            return values


        if not isPSBSupported():
            self.skipTest("PSB period unsupported")

        valid_psb_values = getValidPSBValues()
        # 0 should always be valid, and it's assumed by lldb-server
        self.assertEqual(valid_psb_values[0], 0)

        self.expect("file " + (os.path.join(self.getSourceDir(), "intelpt-trace", "a.out")))
        self.expect("b main")
        self.expect("r")

        # it's enough to test with two valid values
        for psb_period in (valid_psb_values[0], valid_psb_values[-1]):
            # we first test at thread level
            self.traceStartThread(psbPeriod=psb_period)
            self.traceStopThread()

            # we now test at process level
            self.traceStartProcess(psbPeriod=psb_period)
            self.traceStopProcess()

        # we now test invalid values
        self.traceStartThread(psbPeriod=valid_psb_values[-1] + 1, error=True,
            substrs=["Invalid psb_period. Valid values are: 0"])

        # TODO: dump the perf_event_attr.config as part of the upcoming "trace dump info"
        # command and check that the psb period is included there.
