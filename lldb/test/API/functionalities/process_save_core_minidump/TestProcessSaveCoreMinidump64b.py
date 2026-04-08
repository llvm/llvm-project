"""
Test that saved memory regions is byte-wise 1:1 with the live process. Specifically 
that the memory regions that will be populated in the Memory64List are the same byte for byte.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ProcessSaveCoreMinidump64bTestCase(TestBase):
    def verify_minidump(
        self,
        options,
    ):
        """Verify that the minidump is the same byte for byte as the live process."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        core_target = None
        live_proc = target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        try:
            self.assertState(live_proc.GetState(), lldb.eStateStopped)
            error = live_proc.SaveCore(options)
            self.assertTrue(error.Success(), error.GetCString())
            core_target = self.dbg.CreateTarget(None)
            core_proc = target.LoadCore(options.GetOutputFile().fullpath)
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
                actual = live_proc.ReadMemory(
                    start_addr, end_addr - start_addr, actual_process_read_error
                )
                expected_process_read_error = lldb.SBError()
                expected = core_proc.ReadMemory(
                    start_addr, end_addr - start_addr, expected_process_read_error
                )

                # Both processes could fail to read a given memory region, so if they both pass
                # compare, then we'll fail them if the core differs from the live process.
                if (
                    actual_process_read_error.Success()
                    and expected_process_read_error.Success()
                ):
                    self.assertEqual(
                        actual, expected, "Bytes differ between live process and core"
                    )

                # Now we check if the error is the same, error isn't abnormal but they should fail for the same reason
                # Success will be false if they both fail
                self.assertTrue(
                    actual_process_read_error.Success()
                    == expected_process_read_error.Success(),
                    f"Address range {hex(start_addr)} - {hex(end_addr)} failed to read from live process and core for different reasons",
                )
        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))
            if core_target is not None:
                self.assertTrue(self.dbg.DeleteTarget(core_target))

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_minidump_save_style_full(self):
        """Test that a full minidump is the same byte for byte."""
        minidump_path = self.getBuildArtifact("minidump_full_force64b.dmp")
        try:
            options = lldb.SBSaveCoreOptions()
            options.SetOutputFile(lldb.SBFileSpec(minidump_path))
            options.SetStyle(lldb.eSaveCoreFull)
            options.SetPluginName("minidump")
            self.verify_minidump(options)
        finally:
            if os.path.isfile(minidump_path):
                os.unlink(minidump_path)

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_minidump_save_style_mixed_memory(self):
        """Test that a mixed memory minidump is the same byte for byte."""
        minidump_path = self.getBuildArtifact("minidump_mixed_force64b.dmp")
        try:
            options = lldb.SBSaveCoreOptions()
            options.SetOutputFile(lldb.SBFileSpec(minidump_path))
            options.SetStyle(lldb.eSaveCoreDirtyOnly)
            options.SetPluginName("minidump")
            self.verify_minidump(options)
        finally:
            if os.path.isfile(minidump_path):
                os.unlink(minidump_path)
