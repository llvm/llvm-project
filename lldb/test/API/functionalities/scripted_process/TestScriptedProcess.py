"""
Test python scripted process in lldb
"""

import os, shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest

import dummy_scripted_process


class ScriptedProcesTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_python_plugin_package(self):
        """Test that the lldb python module has a `plugins.scripted_process`
        package."""
        self.expect(
            "script import lldb.plugins",
            substrs=["ModuleNotFoundError"],
            matching=False,
        )

        self.expect("script dir(lldb.plugins)", substrs=["scripted_process"])

        self.expect(
            "script import lldb.plugins.scripted_process",
            substrs=["ModuleNotFoundError"],
            matching=False,
        )

        self.expect(
            "script dir(lldb.plugins.scripted_process)", substrs=["ScriptedProcess"]
        )

        self.expect(
            "script from lldb.plugins.scripted_process import ScriptedProcess",
            substrs=["ImportError"],
            matching=False,
        )

        self.expect("script dir(ScriptedProcess)", substrs=["launch"])

    def move_blueprint_to_dsym(self, blueprint_name):
        blueprint_origin_path = os.path.join(self.getSourceDir(), blueprint_name)
        dsym_bundle = self.getBuildArtifact("a.out.dSYM")
        blueprint_destination_path = os.path.join(
            dsym_bundle, "Contents", "Resources", "Python"
        )
        if not os.path.exists(blueprint_destination_path):
            os.mkdir(blueprint_destination_path)

        blueprint_destination_path = os.path.join(
            blueprint_destination_path, "a_out.py"
        )
        shutil.copy(blueprint_origin_path, blueprint_destination_path)

    # No dylib on Windows.
    @skipIfWindows
    def test_missing_methods_scripted_register_context(self):
        """Test that we only instanciate scripted processes if they implement
        all the required abstract methods."""
        self.build()

        os.environ["SKIP_SCRIPTED_PROCESS_LAUNCH"] = "1"

        def cleanup():
            del os.environ["SKIP_SCRIPTED_PROCESS_LAUNCH"]

        self.addTearDownHook(cleanup)

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        log_file = self.getBuildArtifact("script.log")
        self.runCmd("log enable lldb script -f " + log_file)
        self.assertTrue(os.path.isfile(log_file))
        script_path = os.path.join(
            self.getSourceDir(), "missing_methods_scripted_process.py"
        )
        self.runCmd("command script import " + script_path)

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(
            "missing_methods_scripted_process.MissingMethodsScriptedProcess"
        )
        error = lldb.SBError()

        process = target.Launch(launch_info, error)

        self.assertFailure(error)

        with open(log_file, "r") as f:
            log = f.read()

        self.assertIn(
            "Abstract method MissingMethodsScriptedProcess.read_memory_at_address not implemented",
            log,
        )
        self.assertIn(
            "Abstract method MissingMethodsScriptedProcess.is_alive not implemented",
            log,
        )
        self.assertIn(
            "Abstract method MissingMethodsScriptedProcess.get_scripted_thread_plugin not implemented",
            log,
        )

    @skipUnlessDarwin
    def test_invalid_scripted_register_context(self):
        """Test that we can launch an lldb scripted process with an invalid
        Scripted Thread, with invalid register context."""
        self.build()

        os.environ["SKIP_SCRIPTED_PROCESS_LAUNCH"] = "1"

        def cleanup():
            del os.environ["SKIP_SCRIPTED_PROCESS_LAUNCH"]

        self.addTearDownHook(cleanup)

        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.move_blueprint_to_dsym("invalid_scripted_process.py")
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        log_file = self.getBuildArtifact("thread.log")
        self.runCmd("log enable lldb thread -f " + log_file)
        self.assertTrue(os.path.isfile(log_file))

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName("a_out.InvalidScriptedProcess")
        error = lldb.SBError()

        process = target.Launch(launch_info, error)

        self.assertSuccess(error)
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetProcessID(), 666)
        self.assertEqual(process.GetNumThreads(), 0)

        impl = process.GetScriptedImplementation()
        self.assertTrue(impl)
        impl = process.GetScriptedImplementation()
        self.assertTrue(impl)
        impl = process.GetScriptedImplementation()
        self.assertTrue(impl)
        impl = process.GetScriptedImplementation()
        self.assertTrue(impl)

        addr = 0x500000000
        buff = process.ReadMemory(addr, 4, error)
        self.assertEqual(buff, None)
        self.assertTrue(error.Fail())
        self.assertEqual(error.GetCString(), "This is an invalid scripted process!")

        with open(log_file, "r") as f:
            log = f.read()

        self.assertIn("Failed to get scripted thread registers data.", log)

    @skipUnlessDarwin
    def test_scripted_process_and_scripted_thread(self):
        """Test that we can launch an lldb scripted process using the SBAPI,
        check its process ID, read string from memory, check scripted thread
        id, name stop reason and register context.
        """
        self.build()
        target_0 = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target_0, VALID_TARGET)

        os.environ["SKIP_SCRIPTED_PROCESS_LAUNCH"] = "1"

        def cleanup():
            del os.environ["SKIP_SCRIPTED_PROCESS_LAUNCH"]

        self.addTearDownHook(cleanup)

        scripted_process_example_relpath = "dummy_scripted_process.py"
        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), scripted_process_example_relpath)
        )

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(
            "dummy_scripted_process.DummyScriptedProcess"
        )

        error = lldb.SBError()
        process_0 = target_0.Launch(launch_info, error)
        self.assertTrue(process_0 and process_0.IsValid(), PROCESS_IS_VALID)
        self.assertEqual(process_0.GetProcessID(), 42)
        self.assertEqual(process_0.GetNumThreads(), 1)

        py_impl = process_0.GetScriptedImplementation()
        self.assertTrue(py_impl)
        self.assertIsInstance(py_impl, dummy_scripted_process.DummyScriptedProcess)
        self.assertFalse(hasattr(py_impl, "my_super_secret_member"))
        py_impl.my_super_secret_member = 42
        self.assertTrue(hasattr(py_impl, "my_super_secret_member"))
        self.assertEqual(py_impl.my_super_secret_method(), 42)

        # Try reading from target #0 process ...
        addr = 0x500000000
        message = "Hello, target 0"
        buff = process_0.ReadCStringFromMemory(addr, len(message) + 1, error)
        self.assertSuccess(error)
        self.assertEqual(buff, message)

        target_1 = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target_1, VALID_TARGET)

        # We still need to specify a PID when attaching even for scripted processes
        attach_info = lldb.SBAttachInfo(42)
        attach_info.SetProcessPluginName("ScriptedProcess")
        attach_info.SetScriptedProcessClassName(
            "dummy_scripted_process.DummyScriptedProcess"
        )

        error = lldb.SBError()
        process_1 = target_1.Attach(attach_info, error)
        self.assertTrue(process_1 and process_1.IsValid(), PROCESS_IS_VALID)
        self.assertEqual(process_1.GetProcessID(), 42)
        self.assertEqual(process_1.GetNumThreads(), 1)

        # ... then try reading from target #1 process ...
        message = "Hello, target 1"
        buff = process_1.ReadCStringFromMemory(addr, len(message) + 1, error)
        self.assertSuccess(error)
        self.assertEqual(buff, message)

        # ... now, reading again from target #0 process to make sure the call
        # gets dispatched to the right target.
        message = "Hello, target 0"
        buff = process_0.ReadCStringFromMemory(addr, len(message) + 1, error)
        self.assertSuccess(error)
        self.assertEqual(buff, message)

        # Let's write some memory.
        message = "Hello, world!"
        bytes_written = process_0.WriteMemoryAsCString(addr, message, error)
        self.assertSuccess(error)
        self.assertEqual(bytes_written, len(message) + 1)

        # ... and check if that memory was saved properly.
        buff = process_0.ReadCStringFromMemory(addr, len(message) + 1, error)
        self.assertSuccess(error)
        self.assertEqual(buff, message)

        thread = process_0.GetSelectedThread()
        self.assertTrue(thread, "Invalid thread.")
        self.assertEqual(thread.GetThreadID(), 0x19)
        self.assertEqual(thread.GetName(), "DummyScriptedThread.thread-1")
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonTrace)

        self.assertGreater(thread.GetNumFrames(), 0)

        frame = thread.GetFrameAtIndex(0)
        GPRs = None
        register_set = frame.registers  # Returns an SBValueList.
        for regs in register_set:
            if "general purpose" in regs.name.lower():
                GPRs = regs
                break

        self.assertTrue(GPRs, "Invalid General Purpose Registers Set")
        self.assertGreater(GPRs.GetNumChildren(), 0)
        for idx, reg in enumerate(GPRs, start=1):
            if idx > 21:
                break
            self.assertEqual(idx, int(reg.value, 16))

        self.assertTrue(frame.IsArtificial(), "Frame is not artificial")
        pc = frame.GetPCAddress().GetLoadAddress(target_0)
        self.assertEqual(pc, 0x0100001B00)
