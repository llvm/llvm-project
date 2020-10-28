import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceDumpInstructions(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        if 'intel-pt' not in configuration.enabled_plugins:
            self.skipTest("The intel-pt test plugin is not enabled")

    def testErrorMessages(self):
        # We first check the output when there are no targets
        self.expect("thread trace dump instructions",
            substrs=["error: invalid target, create a target using the 'target create' command"],
            error=True)

        # We now check the output when there's a non-running target
        self.expect("target create " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))

        self.expect("thread trace dump instructions",
            substrs=["error: invalid process"],
            error=True)

        # Now we check the output when there's a running target without a trace
        self.expect("b main")
        self.expect("run")

        self.expect("thread trace dump instructions",
            substrs=["error: this thread is not being traced"],
            error=True)

    def testDumpInstructions(self):
        self.expect("trace load -v " + os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
            substrs=["intel-pt"])

        self.expect("thread trace dump instructions",
            substrs=['thread #1: tid = 3842849, total instructions = 1000',
                     'would print 20 instructions from position 0'])

        # We check if we can pass count and offset
        self.expect("thread trace dump instructions --count 5 --start-position 10",
            substrs=['thread #1: tid = 3842849, total instructions = 1000',
                     'would print 5 instructions from position 10'])

        # We check if we can access the thread by index id
        self.expect("thread trace dump instructions 1",
            substrs=['thread #1: tid = 3842849, total instructions = 1000',
                     'would print 20 instructions from position 0'])

        # We check that we get an error when using an invalid thread index id
        self.expect("thread trace dump instructions 10", error=True,
            substrs=['error: no thread with index: "10"'])

    def testDumpInstructionsWithMultipleThreads(self):
        # We load a trace with two threads
        self.expect("trace load -v " + os.path.join(self.getSourceDir(), "intelpt-trace", "trace_2threads.json"))

        # We print the instructions of two threads simultaneously
        self.expect("thread trace dump instructions 1 2",
            substrs=['''thread #1: tid = 3842849, total instructions = 1000
  would print 20 instructions from position 0
thread #2: tid = 3842850, total instructions = 1000
  would print 20 instructions from position 0'''])

        # We use custom --count and --start-position, saving the command to history for later
        ci = self.dbg.GetCommandInterpreter()

        result = lldb.SBCommandReturnObject()
        ci.HandleCommand("thread trace dump instructions 1 2 --count 12 --start-position 5", result, True)
        self.assertIn('''thread #1: tid = 3842849, total instructions = 1000
  would print 12 instructions from position 5
thread #2: tid = 3842850, total instructions = 1000
  would print 12 instructions from position 5''', result.GetOutput())

        # We use a repeat command and ensure the previous count is used and the start-position has moved to the next position
        result = lldb.SBCommandReturnObject()
        ci.HandleCommand("", result)
        self.assertIn('''thread #1: tid = 3842849, total instructions = 1000
  would print 12 instructions from position 17
thread #2: tid = 3842850, total instructions = 1000
  would print 12 instructions from position 17''', result.GetOutput())

        ci.HandleCommand("", result)
        self.assertIn('''thread #1: tid = 3842849, total instructions = 1000
  would print 12 instructions from position 29
thread #2: tid = 3842850, total instructions = 1000
  would print 12 instructions from position 29''', result.GetOutput())
