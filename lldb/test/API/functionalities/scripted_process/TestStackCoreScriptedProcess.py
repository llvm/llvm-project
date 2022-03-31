"""
Test python scripted process in lldb
"""

import os, json, tempfile

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest

class StackCoreScriptedProcesTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    def tearDown(self):
        TestBase.tearDown(self)

    def create_stack_skinny_corefile(self, file):
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(self, "// break here",
                                                                       lldb.SBFileSpec("baz.c"))
        self.assertTrue(process.IsValid(), "Process is invalid.")
        # FIXME: Use SBAPI to save the process corefile.
        self.runCmd("process save-core -s stack  " + file)
        self.assertTrue(os.path.exists(file), "No stack-only corefile found.")
        self.assertTrue(self.dbg.DeleteTarget(target), "Couldn't delete target")

    def get_module_with_name(self, target, name):
        for module in target.modules:
            if name in module.GetFileSpec().GetFilename():
                return module
        return None

    @skipUnlessDarwin
    @skipIfOutOfTreeDebugserver
    def test_launch_scripted_process_stack_frames(self):
        """Test that we can launch an lldb scripted process from the command
        line, check its process ID and read string from memory."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        main_module = self.get_module_with_name(target, 'a.out')
        self.assertTrue(main_module, "Invalid main module.")
        error = target.SetModuleLoadAddress(main_module, 0)
        self.assertSuccess(error, "Reloading main module at offset 0 failed.")

        scripted_dylib = self.get_module_with_name(target, 'libbaz.dylib')
        self.assertTrue(scripted_dylib, "Dynamic library libbaz.dylib not found.")
        self.assertEqual(scripted_dylib.GetObjectFileHeaderAddress().GetLoadAddress(target), 0xffffffffffffffff)

        os.environ['SKIP_SCRIPTED_PROCESS_LAUNCH'] = '1'
        def cleanup():
          del os.environ["SKIP_SCRIPTED_PROCESS_LAUNCH"]
        self.addTearDownHook(cleanup)

        scripted_process_example_relpath = 'stack_core_scripted_process.py'
        self.runCmd("command script import " + os.path.join(self.getSourceDir(),
                                                            scripted_process_example_relpath))

        corefile_process = None
        with tempfile.NamedTemporaryFile() as file:
            self.create_stack_skinny_corefile(file.name)
            corefile_target = self.dbg.CreateTarget(None)
            corefile_process = corefile_target.LoadCore(self.getBuildArtifact(file.name))
        self.assertTrue(corefile_process, PROCESS_IS_VALID)

        structured_data = lldb.SBStructuredData()
        structured_data.SetFromJSON(json.dumps({
            "backing_target_idx" : self.dbg.GetIndexOfTarget(corefile_process.GetTarget()),
            "libbaz_path" : self.getBuildArtifact("libbaz.dylib")
        }))
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName("stack_core_scripted_process.StackCoreScriptedProcess")
        launch_info.SetScriptedProcessDictionary(structured_data)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertSuccess(error)
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetProcessID(), 42)

        self.assertEqual(process.GetNumThreads(), 2)
        thread = process.GetSelectedThread()
        self.assertTrue(thread, "Invalid thread.")
        self.assertEqual(thread.GetName(), "StackCoreScriptedThread.thread-1")

        self.assertTrue(target.triple, "Invalid target triple")
        arch = target.triple.split('-')[0]
        supported_arch = ['x86_64', 'arm64', 'arm64e']
        self.assertIn(arch, supported_arch)
        # When creating a corefile of a arm process, lldb saves the exception
        # that triggers the breakpoint in the LC_NOTES of the corefile, so they
        # can be reloaded with the corefile on the next debug session.
        if arch in 'arm64e':
            self.assertTrue(thread.GetStopReason(), lldb.eStopReasonException)
        # However, it's architecture specific, and corefiles made from intel
        # process don't save any metadata to retrieve to stop reason.
        # To mitigate this, the StackCoreScriptedProcess will report a
        # eStopReasonSignal with a SIGTRAP, mimicking what debugserver does.
        else:
            self.assertTrue(thread.GetStopReason(), lldb.eStopReasonSignal)

        self.assertEqual(thread.GetNumFrames(), 5)
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame, "Invalid frame.")
        func = frame.GetFunction()
        self.assertTrue(func, "Invalid function.")

        self.assertIn("baz", frame.GetFunctionName())
        self.assertEqual(frame.vars.GetSize(), 2)
        self.assertEqual(int(frame.vars.GetFirstValueByName('j').GetValue()), 42 * 42)
        self.assertEqual(int(frame.vars.GetFirstValueByName('k').GetValue()), 42)

        corefile_dylib = self.get_module_with_name(corefile_target, 'libbaz.dylib')
        self.assertTrue(corefile_dylib, "Dynamic library libbaz.dylib not found.")
        scripted_dylib = self.get_module_with_name(target, 'libbaz.dylib')
        self.assertTrue(scripted_dylib, "Dynamic library libbaz.dylib not found.")
        self.assertEqual(scripted_dylib.GetObjectFileHeaderAddress().GetLoadAddress(target),
                         corefile_dylib.GetObjectFileHeaderAddress().GetLoadAddress(target))
