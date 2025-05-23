"""Test for the JITLoaderPython interface"""

import os
import unittest

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


# Find an unmapped region large enough to contain the file specified by path.
def find_empty_memory_region_for_file(process, file_path):
    file_size = os.path.getsize(file_path)
    load_addr = 0
    while True:
        region = lldb.SBMemoryRegionInfo()
        error = process.GetMemoryRegionInfo(load_addr, region)
        if error.Fail():
            return None

        region_base_addr = region.GetRegionBase()
        region_end_addr = region.GetRegionEnd()

        # Abort on bad region
        if region_base_addr >= region_end_addr:
            return None

        load_addr = region_end_addr  # In case we loop set the next load address

        # Skip regions that have any permissions, we are looking for an unmapped
        # region to pretend to load our 'jit.out' file in.
        if region.IsReadable() or region.IsWritable() or region.IsExecutable():
            continue

        if region_base_addr == 0:
            # Don't try to map something at zero, add a small offset.
            region_base_addr += 0x40000

        if region_base_addr + file_size <= region_end_addr:
            return region_base_addr

        if region_end_addr == lldb.LLDB_INVALID_ADDRESS:
            return None


class JITLoaderPythonTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_jit(self):
        """Tests the python JITLoader interface."""
        self.build()

        python_jit_loader_path = self.getSourcePath("jit_loader.py")
        # self.runCmd("log enable -f %s lldb jit" % (logfile))
        self.runCmd(
            "settings set target.process.python-jit-loader-path '%s'"
            % (python_jit_loader_path)
        )

        def cleanup():
            self.runCmd("settings clear target.process.python-jit-loader-path")

        self.addTearDownHook(cleanup)

        exe = self.getBuildArtifact("a.out")
        main_source_spec = lldb.SBFileSpec("main.cpp", False)
        # Launch the process.
        (target, process, thread, bkpt1) = lldbutil.run_to_source_breakpoint(
            self, "// Breakpoint 1", main_source_spec
        )

        jit_exe = self.getBuildArtifact("jit.out")
        jit_exe_addr = find_empty_memory_region_for_file(process, jit_exe)

        frame = thread.GetFrameAtIndex(0)
        # Update the JIT entry path to match our compile jit program so the
        # python JIT loader can get the path to the "jit.out" program.
        result = frame.EvaluateExpression(f'entry.path = "{jit_exe}"')
        self.assertTrue(result.GetError().Success(), "failed to set the jit.out path")
        jit_exe_addr = find_empty_memory_region_for_file(process, jit_exe)
        if jit_exe_addr == None:
            return  # Couldn't find an empty memory range to load jit_exe
        # Update the JIT entry address to the address in a region that is not
        # mapped.
        result = frame.EvaluateExpression(f"entry.address = {jit_exe_addr}")
        self.assertTrue(
            result.GetError().Success(),
            f"failed to set the jit.out address {result.GetError()}",
        )

        lldbutil.continue_to_source_breakpoint(
            self, process, "// Breakpoint 2", main_source_spec
        )
        # The Python JIT loader should have added this module to the target
        # by the time we hit the second breakpoint
        jit_module = target.module["jit.out"]
        self.assertTrue(jit_module.IsValid(), "jit.out module isn't in target")

        lldbutil.continue_to_source_breakpoint(
            self, process, "// Breakpoint 3", main_source_spec
        )

        jit_module = target.module["jit.out"]
        self.assertTrue(jit_module is None, "jit.out module wasn't removed from target")
