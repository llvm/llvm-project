import lldb
import json
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

def find(predicate, seq):
  for item in seq:
    if predicate(item):
      return item

class TestTraceSave(TraceIntelPTTestCaseBase):

    def testErrorMessages(self):
        # We first check the output when there are no targets
        self.expect("trace save",
            substrs=["error: invalid target, create a target using the 'target create' command"],
            error=True)

        # We now check the output when there's a non-running target
        self.expect("target create " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))

        self.expect("trace save",
            substrs=["error: Command requires a current process."],
            error=True)

        # Now we check the output when there's a running target without a trace
        self.expect("b main")
        self.expect("run")

        self.expect("trace save",
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
        self.expect("trace save ",
            substrs=["error: a single path to a directory where the trace bundle will be created is required"],
            error=True)

        # Check the output when saving to an invalid directory
        self.expect("trace save /",
            substrs=["error: couldn't write to the file"],
            error=True)

    def testSaveWhenNotLiveTrace(self):
        self.expect("trace load -v " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
            substrs=["intel-pt"])

        # Check the output when not doing live tracing
        self.expect("trace save " +
            os.path.join(self.getBuildDir(), "intelpt-trace", "trace_not_live_dir"))

    def testSaveMultiCpuTrace(self):
        '''
            This test starts a per-cpu tracing session, then saves the session to disk, and
            finally it loads it again.
        '''
        self.skipIfPerCpuTracingIsNotSupported()

        self.expect("target create " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")
        self.expect("process trace start --per-cpu-tracing")
        self.expect("b 7")

        output_dir = os.path.join(self.getBuildDir(), "intelpt-trace", "trace_save")
        self.expect("trace save " + output_dir)

        def checkSessionBundle(session_file_path):
            with open(session_file_path) as session_file:
                session = json.load(session_file)
                # We expect tsc conversion info
                self.assertTrue("tscPerfZeroConversion" in session)
                # We expect at least one cpu
                self.assertGreater(len(session["cpus"]), 0)

                # We expect the required trace files to be created
                for cpu in session["cpus"]:
                    cpu_files_prefix = os.path.join(output_dir, "cpus", str(cpu["id"]))
                    self.assertTrue(os.path.exists(cpu_files_prefix + ".intelpt_trace"))
                    self.assertTrue(os.path.exists(cpu_files_prefix + ".perf_context_switch_trace"))

                # We expect at least one one process
                self.assertGreater(len(session["processes"]), 0)
                for process in session["processes"]:
                    # We expect at least one thread
                    self.assertGreater(len(process["threads"]), 0)
                    # We don't expect thread traces
                    for thread in process["threads"]:
                        self.assertTrue(("iptTrace" not in thread) or (thread["iptTrace"] is None))

        original_trace_session_file = os.path.join(output_dir, "trace.json")
        checkSessionBundle(original_trace_session_file)

        output_dir = os.path.join(self.getBuildDir(), "intelpt-trace", "trace_save")
        self.expect("trace load " + os.path.join(output_dir, "trace.json"))
        output_copy_dir = os.path.join(self.getBuildDir(), "intelpt-trace", "copy_trace_save")
        self.expect("trace save " + output_copy_dir)

        # We now check that the new bundle is correct on its own
        copied_trace_session_file = os.path.join(output_copy_dir, "trace.json")
        checkSessionBundle(copied_trace_session_file)

        # We finally check that the new bundle has the same information as the original one
        with open(original_trace_session_file) as original_file:
            original = json.load(original_file)
            with open(copied_trace_session_file) as copy_file:
                copy = json.load(copy_file)

                self.assertEqual(len(original["processes"]), len(copy["processes"]))

                for process in original["processes"]:
                    copied_process = find(lambda proc : proc["pid"] == process["pid"], copy["processes"])
                    self.assertTrue(copied_process is not None)

                    for thread in process["threads"]:
                        copied_thread = find(lambda thr : thr["tid"] == thread["tid"], copied_process["threads"])
                        self.assertTrue(copied_thread is not None)

                for cpu in original["cpus"]:
                    copied_cpu = find(lambda cor : cor["id"] == cpu["id"], copy["cpus"])
                    self.assertTrue(copied_cpu is not None)

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
        self.expect("trace save " +
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
