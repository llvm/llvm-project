"""
Test Debugger APIs.
"""

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DebuggerAPITestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_debugger_api_boundary_condition(self):
        """Exercise SBDebugger APIs with boundary conditions."""
        self.dbg.HandleCommand(None)
        self.dbg.SetDefaultArchitecture(None)
        self.dbg.GetScriptingLanguage(None)
        self.dbg.CreateTarget(None)
        self.dbg.CreateTarget(None, None, None, True, lldb.SBError())
        self.dbg.CreateTargetWithFileAndTargetTriple(None, None)
        self.dbg.CreateTargetWithFileAndArch(None, None)
        self.dbg.FindTargetWithFileAndArch(None, None)
        self.dbg.SetInternalVariable(None, None, None)
        self.dbg.GetInternalVariableValue(None, None)
        # FIXME (filcab): We must first allow for the swig bindings to know if
        # a Python callback is set. (Check python-typemaps.swig)
        # self.dbg.SetLoggingCallback(None)
        self.dbg.SetPrompt(None)
        self.dbg.SetCurrentPlatform(None)
        self.dbg.SetCurrentPlatformSDKRoot(None)
        
        fresh_dbg = lldb.SBDebugger()
        self.assertEquals(len(fresh_dbg), 0)

    def test_debugger_delete_invalid_target(self):
        """SBDebugger.DeleteTarget() should not crash LLDB given and invalid target."""
        target = lldb.SBTarget()
        self.assertFalse(target.IsValid())
        self.dbg.DeleteTarget(target)

    def test_debugger_internal_variables(self):
        """Ensure that SBDebugger reachs the same instance of properties
           regardless CommandInterpreter's context initialization"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        property_name = "target.process.memory-cache-line-size"

        def get_cache_line_size():
            value_list = lldb.SBStringList()
            value_list = self.dbg.GetInternalVariableValue(property_name,
                                                           self.dbg.GetInstanceName())

            self.assertEqual(value_list.GetSize(), 1)
            try:
                return int(value_list.GetStringAtIndex(0))
            except ValueError as error:
                self.fail("Value is not a number: " + error)

        # Get global property value while there are no processes.
        global_cache_line_size = get_cache_line_size()

        # Run a process via SB interface. CommandInterpreter's execution context
        # remains empty.
        error = lldb.SBError()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetLaunchFlags(lldb.eLaunchFlagStopAtEntry)
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # This should change the value of a process's local property.
        new_cache_line_size = global_cache_line_size + 512
        error = self.dbg.SetInternalVariable(property_name,
                                             str(new_cache_line_size),
                                             self.dbg.GetInstanceName())
        self.assertSuccess(error,
                           property_name + " value was changed successfully")

        # Check that it was set actually.
        self.assertEqual(get_cache_line_size(), new_cache_line_size)

        # Run any command to initialize CommandInterpreter's execution context.
        self.runCmd("target list")

        # Test the local property again, is it set to new_cache_line_size?
        self.assertEqual(get_cache_line_size(), new_cache_line_size)

    def test_CreateTarget_platform(self):
        exe = self.getBuildArtifact("a.out")
        self.yaml2obj("elf.yaml", exe)
        error = lldb.SBError()
        target1 = self.dbg.CreateTarget(exe, None, "remote-linux",
                False, error)
        self.assertSuccess(error)
        platform1 = target1.GetPlatform()
        platform1.SetWorkingDirectory("/foo/bar")

        # Reuse a platform if it matches the currently selected one...
        target2 = self.dbg.CreateTarget(exe, None, "remote-linux",
                False, error)
        self.assertSuccess(error)
        platform2 = target2.GetPlatform()
        self.assertTrue(platform2.GetWorkingDirectory().endswith("bar"),
                platform2.GetWorkingDirectory())

        # ... but create a new one if it doesn't.
        self.dbg.SetSelectedPlatform(lldb.SBPlatform("remote-windows"))
        target3 = self.dbg.CreateTarget(exe, None, "remote-linux",
                False, error)
        self.assertSuccess(error)
        platform3 = target3.GetPlatform()
        self.assertIsNone(platform3.GetWorkingDirectory())

    def test_CreateTarget_arch(self):
        exe = self.getBuildArtifact("a.out")
        if lldbplatformutil.getHostPlatform() == 'linux':
            self.yaml2obj("macho.yaml", exe)
            arch = "x86_64-apple-macosx"
            expected_platform = "remote-macosx"
        else:
            self.yaml2obj("elf.yaml", exe)
            arch = "x86_64-pc-linux"
            expected_platform = "remote-linux"

        fbsd = lldb.SBPlatform("remote-freebsd")
        self.dbg.SetSelectedPlatform(fbsd)

        error = lldb.SBError()
        target1 = self.dbg.CreateTarget(exe, arch, None, False, error)
        self.assertSuccess(error)
        platform1 = target1.GetPlatform()
        self.assertEqual(platform1.GetName(), expected_platform)
        platform1.SetWorkingDirectory("/foo/bar")

        # Reuse a platform even if it is not currently selected.
        self.dbg.SetSelectedPlatform(fbsd)
        target2 = self.dbg.CreateTarget(exe, arch, None, False, error)
        self.assertSuccess(error)
        platform2 = target2.GetPlatform()
        self.assertEqual(platform2.GetName(), expected_platform)
        self.assertTrue(platform2.GetWorkingDirectory().endswith("bar"),
                platform2.GetWorkingDirectory())
