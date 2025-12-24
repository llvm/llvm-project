import re
import uuid

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class SBTargetExtensionsTestCase(TestBase):

    def test_equality(self):
        """Test the equality operator for SBTarget."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        self.assertEqual(target, target)
        self.assertNotEqual(target, lldb.SBTarget())

    def test_module_access(self):
        """Test the module access extension properties and methods."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        self.assertTrue(len(target.modules) > 0)
        module = target.module[0]
        self.assertTrue(module.IsValid())

        self.assertEqual(target.module["a.out"], module)
        self.assertEqual(target.module[module.file.fullpath], module)

        # UUID strings on Linux might not be standard UUIDs (they are Build IDs).
        # We try to convert, but if it fails, we skip the UUID object check.
        uuid_str = module.GetUUIDString()
        if uuid_str:
            try:
                uuid_obj = uuid.UUID(uuid_str)
                self.assertEqual(target.module[uuid_obj], module)
            except ValueError:
                # The UUID string wasn't a standard UUID format, which is fine on Linux.
                pass

        self.assertEqual(len(target.module[re.compile("a.out")]), 1)
        self.assertEqual(target.module[re.compile("a.out")][0], module)

    def test_process_creation(self):
        """Test process creation via extensions."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        self.assertFalse(target.process.IsValid())

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid())

        # SBProcess objects don't support direct equality (==), compare IDs.
        self.assertEqual(target.process.GetProcessID(), process.GetProcessID())

    def test_breakpoints(self):
        """Test breakpoint access via extensions."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        breakpoint = target.BreakpointCreateBySourceRegex("Set breakpoint here", lldb.SBFileSpec("main.c"))
        self.assertTrue(breakpoint.IsValid())

        self.assertEqual(target.num_breakpoints, 1)
        self.assertEqual(len(target.breakpoints), 1)

        # target.breakpoint[i] uses INDEX, not ID.
        self.assertEqual(target.breakpoint[0].GetID(), target.breakpoints[0].GetID())

        # To verify ID lookup works via the standard API:
        self.assertEqual(target.FindBreakpointByID(breakpoint.GetID()).GetID(), breakpoint.GetID())

    def test_watchpoints(self):
        """Test watchpoint access via extensions."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        # 1. Set a breakpoint so the process stops and stays alive.
        breakpoint = target.BreakpointCreateBySourceRegex("Set breakpoint here", lldb.SBFileSpec("main.c"))
        self.assertTrue(breakpoint.IsValid())

        # 2. Launch the process.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid())

        # 3. Ensure we are stopped.
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        variables = target.FindGlobalVariables("g_var", 1)
        self.assertTrue(variables.GetSize() > 0)

        global_variable = variables.GetValueAtIndex(0)
        error = lldb.SBError()

        # 4. Now we can set the watchpoint.
        watchpoint = target.WatchAddress(global_variable.GetLoadAddress(), 4, False, True, error)
        self.assertTrue(error.Success(), f"Watchpoint failed: {error.GetCString()}")

        self.assertTrue(target.num_watchpoints > 0)
        self.assertEqual(len(target.watchpoints), target.num_watchpoints)

        self.assertEqual(target.watchpoint[0].GetID(), target.watchpoints[0].GetID())
        self.assertEqual(target.watchpoint[0].GetID(), watchpoint.GetID())

    def test_other_properties(self):
        """Test miscellaneous properties of SBTarget."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        self.assertTrue(target.executable.IsValid())

        self.assertEqual(target.debugger.GetID(), self.dbg.GetID())

        self.assertTrue(target.broadcaster.IsValid())
        self.assertIn(target.byte_order, [lldb.eByteOrderLittle, lldb.eByteOrderBig, lldb.eByteOrderInvalid])
        self.assertTrue(target.addr_size > 0)
        self.assertIsNotNone(target.triple)
        self.assertIsNotNone(target.arch_name)

        self.assertTrue(target.data_byte_size > 0)
        self.assertTrue(target.code_byte_size > 0)

        self.assertTrue(target.platform.IsValid())