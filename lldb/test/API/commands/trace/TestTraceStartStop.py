import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceLoad(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        if 'intel-pt' not in configuration.enabled_plugins:
            self.skipTest("The intel-pt test plugin is not enabled")

    def expectGenericHelpMessageForStartCommand(self):
        self.expect("help thread trace start",
            substrs=["Syntax: thread trace start [<trace-options>]"])

    def testStartStopSessionFileThreads(self):
        # it should fail for processes from json session files
        self.expect("trace load -v " + os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"))
        self.expect("thread trace start", error=True,
            substrs=["error: Tracing is not supported. Can't trace a non-live process"])

        # the help command should be the generic one, as it's not a live process
        self.expectGenericHelpMessageForStartCommand()

        # this should fail because 'trace stop' is not yet implemented
        self.expect("thread trace stop", error=True,
            substrs=["error: Failed stopping thread 3842849"])

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testStartStopLiveThreads(self):
        # The help command should be the generic one if there's no process running
        self.expectGenericHelpMessageForStartCommand()

        self.expect("thread trace start", error=True,
            substrs=["error: Process not available"])

        self.expect("file " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")

        self.expect("thread trace start", error=True,
            substrs=["error: Process not available"])

        # The help command should be the generic one if there's still no process running
        self.expectGenericHelpMessageForStartCommand()

        self.expect("r")

        # the help command should be the intel-pt one now
        self.expect("help thread trace start",
            substrs=["Start tracing one or more threads with intel-pt.",
                     "Syntax: thread trace start [<thread-index> <thread-index> ...] [<intel-pt-options>]"])

        self.expect("thread trace start",
            patterns=["would trace tid .* with size_in_kb 4 and custom_config 0"])

        self.expect("thread trace start --size 20 --custom-config 1",
            patterns=["would trace tid .* with size_in_kb 20 and custom_config 1"])

        # This fails because "trace stop" is not yet implemented.
        self.expect("thread trace stop", error=True,
            substrs=["error: Process is not being traced"])

        self.expect("c")
        # Now the process has finished, so the commands should fail
        self.expect("thread trace start", error=True,
            substrs=["error: Process must be launched"])

        self.expect("thread trace stop", error=True,
            substrs=["error: Process must be launched"])
