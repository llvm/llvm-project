"""
Test Process::FindInMemory.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import configuration
from lldbsuite.test import lldbutil
from address_ranges_helper import *


class FindInMemoryTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def BuildAndRunToBreakpoint(self) -> ProcessInfo:
        live_pi = ProcessInfo()
        self.build()
        (
            live_pi.target,
            live_pi.process,
            live_pi.thread,
            live_pi.bp,
        ) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(live_pi.bp.IsValid())
        live_pi.frame = live_pi.thread.GetFrameAtIndex(0)
        return live_pi

    def CreateCoreProcess(self, plugin_name: str, exe_name: str = None) -> ProcessInfo:
        core_path = self.getBuildArtifact("a.out.core")
        exe_path = None
        if plugin_name == "elf":
            self.assertTrue(exe_name)
            exe_path = os.path.join(configuration.test_src_root, self.mydir, exe_name)
            self.assertTrue(os.path.isfile(exe_path))
            self.yaml2obj("linux-x86_64.core.yaml", core_path)
        else:
            save_core_command = f"process save-core --plugin-name={plugin_name} --style=modified '{core_path}'"
            self.runCmd(save_core_command)

        self.assertTrue(os.path.isfile(core_path))
        core_pi = ProcessInfo()
        core_pi.target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(core_pi.target.IsValid())
        core_pi.process = core_pi.target.LoadCore(core_path)
        self.assertTrue(core_pi.process.IsValid())
        core_pi.thread = core_pi.process.GetSelectedThread()
        core_pi.frame = core_pi.thread.GetFrameAtIndex(0)
        self.assertTrue(core_pi.process, PROCESS_IS_VALID)

        return core_pi

    def GetVariable(self, pi: ProcessInfo, var_name: str) -> lldb.SBValue:
        var = pi.frame.EvaluateExpression(var_name)
        self.assertTrue(var.IsValid())
        region = lldb.SBMemoryRegionInfo()
        self.assertTrue(
            pi.process.GetMemoryRegionInfo(var.GetValueAsUnsigned(), region)
        )
        self.assertTrue(region.IsReadable(), f"Invalid region {region} for {var}")
        return var

    def test_minidump_find_ranges_in_memory_core_heap_ok(self):
        """
        Make sure a match exists in the core's heap memory and the right address ranges are provided.
        """
        _ = self.BuildAndRunToBreakpoint()
        core_pi = self.CreateCoreProcess("minidump")
        core_var = self.GetVariable(core_pi, "heap_pointer1")

        error = lldb.SBError()
        ranges = GetHeapRanges(self, core_pi)
        self.assertTrue(ranges.GetSize() > 0)

        addr = core_pi.process.FindInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            ranges[0],
            1,
            error,
        )
        self.assertSuccess(error)
        self.assertNotEqual(
            addr,
            lldb.LLDB_INVALID_ADDRESS,
            f"No matches found for {core_var} in {ranges[0]}",
        )

        matches = core_pi.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            ranges,
            1,
            10,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(
            matches.GetSize(),
            2,
            f"Failed to find all matches for {core_var} in {ranges}",
        )

    def test_minidump_find_ranges_in_memory_core_stack_ok(self):
        """
        Make sure a match exists in the core's heap memory and the right address ranges are provided.
        """
        _ = self.BuildAndRunToBreakpoint()
        core_pi = self.CreateCoreProcess("minidump")
        core_var = core_pi.frame.FindVariable("&stack_pointer")

        error = lldb.SBError()
        ranges = GetStackRanges(self, core_pi)
        self.assertTrue(ranges.GetSize() > 0)

        addr = core_pi.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            ranges[0],
            1,
            error,
        )
        self.assertSuccess(error)
        self.assertNotEqual(
            addr,
            lldb.LLDB_INVALID_ADDRESS,
            f"No matches found for {core_var} in {range}",
        )

        matches = core_pi.process.FindRangesInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            ranges,
            1,
            10,
            error,
        )
        self.assertSuccess(error)
        self.assertEqual(
            matches.GetSize(), 1, f"Failed to find matche for {core_var} in {ranges}"
        )

    @skipIf(archs=no_match(["arm64", "arm64e", "aarch64"]))
    @skipUnlessDarwin
    def test_macho_find_ranges_in_memory_core_heap_ok(self):
        """
        Make sure a match exists in the core's heap memory and the right address ranges are provided.
        """
        _ = self.BuildAndRunToBreakpoint()
        core_pi = self.CreateCoreProcess("mach-o")
        core_var = self.GetVariable(core_pi, "heap_pointer1")

        error = lldb.SBError()
        ranges = GetHeapRanges(self, core_pi)
        self.assertTrue(ranges.GetSize() > 0)

        self.assertTrue(ranges.GetSize() > 0)
        addr = core_pi.process.FindInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            ranges[0],
            1,
            error,
        )
        self.assertSuccess(error)
        self.assertNotEqual(
            addr,
            lldb.LLDB_INVALID_ADDRESS,
            f"No matches found for {core_var} in {ranges[0]}",
        )

        matches = core_pi.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            ranges,
            1,
            10,
            error,
        )
        self.assertSuccess(error)
        self.assertEqual(
            matches.GetSize(),
            2,
            f"Failed to find all matches for {core_var} in {ranges}",
        )

    @skipIf(archs=no_match(["arm64", "arm64e", "aarch64"]))
    @skipUnlessDarwin
    def test_macho_find_ranges_in_memory_core_stack_ok(self):
        """
        Make sure a match exists in the core's heap memory and the right address ranges are provided.
        """
        _ = self.BuildAndRunToBreakpoint()
        core_pi = self.CreateCoreProcess("mach-o")
        core_var = core_pi.frame.FindVariable("&stack_pointer")

        error = lldb.SBError()
        ranges = GetStackRanges(self, core_pi)
        self.assertTrue(ranges.GetSize() > 0)

        addr = core_pi.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            ranges[0],
            1,
            error,
        )
        self.assertSuccess(error)
        self.assertNotEqual(
            addr,
            lldb.LLDB_INVALID_ADDRESS,
            f"No matches found for {core_var} in {range}",
        )

        matches = core_pi.process.FindRangesInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            ranges,
            1,
            10,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(
            matches.GetSize(), 1, f"Failed to find matche for {core_var} in {ranges}"
        )

    @skipIf(oslist=no_match(["linux"]), archs=no_match(["x86_64"]))
    def test_elf_find_ranges_in_memory_core_heap_ok(self):
        """
        Make sure a match exists in the core's heap memory and the right address ranges are provided.
        """
        core_pi = self.CreateCoreProcess("elf", "linux-x86_64")
        core_var = self.GetVariable(core_pi, "heap_pointer1")

        error = lldb.SBError()
        ranges = GetHeapRanges(self, core_pi)
        self.assertTrue(ranges.GetSize() > 0)

        self.assertTrue(ranges.GetSize() > 0)
        addr = core_pi.process.FindInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            ranges[0],
            1,
            error,
        )
        self.assertSuccess(error)
        self.assertNotEqual(
            addr,
            lldb.LLDB_INVALID_ADDRESS,
            f"No matches found for {core_var} in {ranges[0]}",
        )

        matches = core_pi.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            ranges,
            1,
            10,
            error,
        )
        self.assertSuccess(error)
        self.assertEqual(
            matches.GetSize(),
            2,
            f"Failed to find all matches for {core_var} in {ranges}",
        )

    @skipIf(oslist=no_match(["linux"]), archs=no_match(["x86_64"]))
    def test_elf_find_ranges_in_memory_core_stack_ok(self):
        """
        Make sure a match exists in the core's heap memory and the right address ranges are provided.
        """
        core_pi = self.CreateCoreProcess("elf", "linux-x86_64")
        core_var = self.GetVariable(core_pi, "&stack_pointer")

        error = lldb.SBError()
        ranges = GetStackRanges(self, core_pi)
        self.assertTrue(ranges.GetSize() > 0)

        addr = core_pi.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            ranges[0],
            1,
            error,
        )
        self.assertSuccess(error)
        self.assertNotEqual(
            addr,
            lldb.LLDB_INVALID_ADDRESS,
            f"No matches found for {core_var} in {range}",
        )

        matches = core_pi.process.FindRangesInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            ranges,
            1,
            10,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(
            matches.GetSize(), 1, f"Failed to find matche for {core_var} in {ranges}"
        )
