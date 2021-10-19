import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceSave(TraceIntelPTTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    def testErrorMessages(self):
        # We first check the output when there are no targets
        self.expect("process trace save",
            substrs=["error: invalid target, create a target using the 'target create' command"],
            error=True)

        # We now check the output when there's a non-running target
        self.expect("target create " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))

        self.expect("process trace save",
            substrs=["error: invalid process"],
            error=True)

        # Now we check the output when there's a running target without a trace
        self.expect("b main")
        self.expect("run")

        self.expect("process trace save",
            substrs=["error: Process is not being traced"],
            error=True)

    def testSaveToInvalidDir(self):
        self.expect("target create " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")
        self.expect("thread trace start")
        self.expect("n")

        # Check the output when saving without providing the directory argument
        self.expect("process trace save -d",
            substrs=["error: last option requires an argument"],
            error=True)

        # Check the output when saving to an invalid directory
        self.expect("process trace save -d /",
            substrs=["error: couldn't write to the file"],
            error=True)

    def testSaveWhenNotLiveTrace(self):
        self.expect("trace load -v " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
            substrs=["intel-pt"])

        # Check the output when not doing live tracing
        self.expect("process trace save -d " +
            os.path.join(self.getBuildDir(), "intelpt-trace", "trace_not_live_dir"),
            substrs=["error: Saving a trace requires a live process."],
            error=True)


    def testSaveTrace(self):
        self.expect("target create " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")
        self.expect("thread trace start")
        self.expect("b 7")

        ci = self.dbg.GetCommandInterpreter()
        res = lldb.SBCommandReturnObject()

        ci.HandleCommand("thread trace dump instructions -c 10 --forwards", res)
        self.assertEqual(res.Succeeded(), True)
        first_ten_instructions = res.GetOutput()

        ci.HandleCommand("thread trace dump instructions -c 10", res)
        self.assertEqual(res.Succeeded(), True)
        last_ten_instructions = res.GetOutput()

        # Now, save the trace to <trace_copy_dir>
        self.expect("process trace save -d " +
            os.path.join(self.getBuildDir(), "intelpt-trace", "trace_copy_dir"))

        # Load the trace just saved
        self.expect("trace load -v " +
            os.path.join(self.getBuildDir(), "intelpt-trace", "trace_copy_dir", "trace.json"),
            substrs=["intel-pt"])

        # Compare with instructions saved at the first time
        ci.HandleCommand("thread trace dump instructions -c 10 --forwards", res)
        self.assertEqual(res.Succeeded(), True)
        self.assertEqual(res.GetOutput(), first_ten_instructions)

        ci.HandleCommand("thread trace dump instructions -c 10", res)
        self.assertEqual(res.Succeeded(), True)
        self.assertEqual(res.GetOutput(), last_ten_instructions)
