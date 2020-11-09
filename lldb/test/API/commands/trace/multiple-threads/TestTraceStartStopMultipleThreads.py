import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

ADDRESS_REGEX = '0x[0-9a-fA-F]*'

class TestTraceStartStopMultipleThreads(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        if 'intel-pt' not in configuration.enabled_plugins:
            self.skipTest("The intel-pt test plugin is not enabled")

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testStartMultipleLiveThreads(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("b 6")
        self.expect("b 11")

        self.expect("r")
        self.expect("proce trace start")


        # We'll see here the first thread
        self.expect("continue")
        self.expect("thread trace dump instructions", substrs=['main.cpp:9'])
        
        # We'll see here the second thread
        self.expect("continue")
        self.expect("thread trace dump instructions", substrs=['main.cpp:4'])

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testStartMultipleLiveThreadsWithStops(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("b 6")
        self.expect("b 11")

        self.expect("r")
        self.expect("process trace start")


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

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testStartMultipleLiveThreadsWithStops(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("b 6")
        self.expect("b 11")

        self.expect("r")
        self.expect("process trace start")

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
    def testStartMultipleLiveThreadsWithSmallTotalLimit(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.expect("b main")
        self.expect("r")
        # trace the entire process with enough total size for 1 thread trace
        self.expect("process trace start -l 5000")
        # we get the stop event when trace 2 appears and can't be traced
        self.expect("c", substrs=['Thread', "can't be traced"])
        # we get the stop event when trace 3 appears and can't be traced
        self.expect("c", substrs=['Thread', "can't be traced"])
