"""
Test saving a minidumps with the force 64b flag, and evaluate that every
saved memory region is byte-wise 1:1 with the live process.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

# Constant from MinidumpFileBuilder.h, this forces 64b for non threads
FORCE_64B = "force_64b"

class ProcessSaveCoreMinidump64bTestCase(TestBase):

    def verify_minidump(
        self,
        core_proc,
        live_proc,
        options,
    ):
        """Verify that the minidump is the same byte for byte as the live process."""
        # Get the memory regions we saved off in this core, we can't compare to the core
        # because we pull from /proc/pid/maps, so even ranges that don't get mapped in will show up
        # as ranges in the minidump.
        #
        # Instead, we have an API that returns to us the number of regions we planned to save from the live process
        # and we compare those
        memory_regions_to_compare = options.GetMemoryRegionsToSave()

        for region in memory_regions_to_compare:
            start_addr = region.GetRegionBase()
            end_addr = region.GetRegionEnd()
            actual_process_read_error = lldb.SBError()
            actual = live_proc.ReadMemory(start_addr, end_addr - start_addr, actual_process_read_error)
            expected_process_read_error = lldb.SBError()
            expected = core_proc.ReadMemory(start_addr, end_addr - start_addr, expected_process_read_error)

            # Both processes could fail to read a given memory region, so if they both pass
            # compare, then we'll fail them if the core differs from the live process.
            if (actual_process_read_error.Success() and expected_process_read_error.Success()):
                self.assertEqual(actual, expected, "Bytes differ between live process and core")

            # Now we check if the error is the same, error isn't abnormal but they should fail for the same reason
            self.assertTrue(
                (actual_process_read_error.Success() and expected_process_read_error.Success()) or
                (actual_process_read_error.Fail() and expected_process_read_error.Fail())
            )

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_minidump_save_style_full(self):
        """Test that a full minidump is the same byte for byte."""

        self.build()
        exe = self.getBuildArtifact("a.out")
        minidump_path = self.getBuildArtifact("minidump_full_force64b.dmp")

        try:
            target = self.dbg.CreateTarget(exe)
            live_process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(live_process.GetState(), lldb.eStateStopped)
            options = lldb.SBSaveCoreOptions()

            options.SetOutputFile(lldb.SBFileSpec(minidump_path))
            options.SetStyle(lldb.eSaveCoreFull)
            options.SetPluginName("minidump")
            options.SetProcess(live_process)
            options.AddFlag(FORCE_64B)

            error = live_process.SaveCore(options)
            self.assertTrue(error.Success(), error.GetCString())

            target = self.dbg.CreateTarget(None)
            core_proc = target.LoadCore(minidump_path)

            self.verify_minidump(core_proc, live_process, options)
        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))
            if os.path.isfile(minidump_path):
                os.unlink(minidump_path)
