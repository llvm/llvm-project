"""
Test python scripted process in lldb
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest


class ScriptedProcesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    def tearDown(self):
        TestBase.tearDown(self)

    def test_python_plugin_package(self):
        """Test that the lldb python module has a `plugins.scripted_process`
        package."""
        self.expect('script import lldb.plugins',
                    substrs=["ModuleNotFoundError"], matching=False)

        self.expect('script dir(lldb.plugins)',
                    substrs=["scripted_process"])

        self.expect('script import lldb.plugins.scripted_process',
                    substrs=["ModuleNotFoundError"], matching=False)

        self.expect('script dir(lldb.plugins.scripted_process)',
                    substrs=["ScriptedProcess"])

        self.expect('script from lldb.plugins.scripted_process import ScriptedProcess',
                    substrs=["ImportError"], matching=False)

        self.expect('script dir(ScriptedProcess)',
                    substrs=["launch"])

    def test_launch_scripted_process_sbapi(self):
        """Test that we can launch an lldb scripted process using the SBAPI,
        check its process ID and read string from memory."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        scripted_process_example_relpath = ['..','..','..','..','examples','python','scripted_process','my_scripted_process.py']
        os.environ['SKIP_SCRIPTED_PROCESS_LAUNCH'] = '1'
        self.runCmd("command script import " + os.path.join(self.getSourceDir(),
                                                            *scripted_process_example_relpath))

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName("my_scripted_process.MyScriptedProcess")

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process and process.IsValid(), PROCESS_IS_VALID)
        self.assertEqual(process.GetProcessID(), 42)

        hello_world = "Hello, world!"
        memory_read = process.ReadCStringFromMemory(0x50000000000,
                                                    len(hello_world) + 1, # NULL byte
                                                    error)

        self.assertTrue(error.Success(), "Failed to read memory from scripted process.")
        self.assertEqual(hello_world, memory_read)

        self.assertEqual(process.GetNumThreads(), 1)

        thread = process.GetSelectedThread()
        self.assertTrue(thread, "Invalid thread.")
        self.assertEqual(thread.GetThreadID(), 0x19)
        self.assertEqual(thread.GetName(), "MyScriptedThread.thread-1")
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)

        self.assertGreater(thread.GetNumFrames(), 0)

        frame = thread.GetFrameAtIndex(0)
        register_set = frame.registers # Returns an SBValueList.
        for regs in register_set:
            if 'GPR' in regs.name:
                registers  = regs
                break

        self.assertTrue(registers, "Invalid General Purpose Registers Set")
        self.assertEqual(registers.GetNumChildren(), 21)
        for idx, reg in enumerate(registers, start=1):
            self.assertEqual(idx, int(reg.value, 16))

    def test_launch_scripted_process_cli(self):
        """Test that we can launch an lldb scripted process from the command
        line, check its process ID and read string from memory."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        scripted_process_example_relpath = ['..','..','..','..','examples','python','scripted_process','my_scripted_process.py']
        self.runCmd("command script import " + os.path.join(self.getSourceDir(),
                                                            *scripted_process_example_relpath))

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetProcessID(), 42)

        error = lldb.SBError()
        hello_world = "Hello, world!"
        memory_read = process.ReadCStringFromMemory(0x50000000000,
                                                    len(hello_world) + 1, # NULL byte
                                                    error)

        self.assertTrue(error.Success(), "Failed to read memory from scripted process.")
        self.assertEqual(hello_world, memory_read)
