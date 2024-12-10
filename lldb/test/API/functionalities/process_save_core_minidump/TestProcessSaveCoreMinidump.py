"""
Test saving a mini dump.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ProcessSaveCoreMinidumpTestCase(TestBase):
    def verify_core_file(
        self,
        core_path,
        expected_pid,
        expected_modules,
        expected_threads,
        stacks_to_sps_map,
        stacks_to_registers_map,
    ):
        # To verify, we'll launch with the mini dump
        target = self.dbg.CreateTarget(None)
        process = target.LoadCore(core_path)

        # check if the core is in desired state
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertTrue(process.GetProcessInfo().IsValid())
        self.assertEqual(process.GetProcessInfo().GetProcessID(), expected_pid)
        self.assertNotEqual(target.GetTriple().find("linux"), -1)
        self.assertTrue(target.GetNumModules(), len(expected_modules))
        self.assertEqual(process.GetNumThreads(), len(expected_threads))

        for module, expected in zip(target.modules, expected_modules):
            self.assertTrue(module.IsValid())
            module_file_name = module.GetFileSpec().GetFilename()
            expected_file_name = expected.GetFileSpec().GetFilename()
            # skip kernel virtual dynamic shared objects
            if "vdso" in expected_file_name:
                continue
            self.assertEqual(module_file_name, expected_file_name)
            self.assertEqual(module.GetUUIDString(), expected.GetUUIDString())

        red_zone = process.GetTarget().GetStackRedZoneSize()
        for thread_idx in range(process.GetNumThreads()):
            thread = process.GetThreadAtIndex(thread_idx)
            self.assertTrue(thread.IsValid())
            thread_id = thread.GetThreadID()
            self.assertIn(thread_id, expected_threads)
            frame = thread.GetFrameAtIndex(0)
            sp_region = lldb.SBMemoryRegionInfo()
            sp = frame.GetSP()
            err = process.GetMemoryRegionInfo(sp, sp_region)
            self.assertTrue(err.Success(), err.GetCString())
            error = lldb.SBError()
            # Ensure thread_id is in the saved map
            self.assertIn(thread_id, stacks_to_sps_map)
            # Ensure the SP is correct
            self.assertEqual(stacks_to_sps_map[thread_id], sp)
            # Try to read at the end of the stack red zone and succeed
            process.ReadMemory(sp - red_zone, 1, error)
            self.assertTrue(error.Success(), error.GetCString())
            # Try to read just past the red zone and fail
            process.ReadMemory(sp - red_zone - 1, 1, error)
            self.assertTrue(error.Fail(), "No failure when reading past the red zone")
            # Verify the registers are the same
            self.assertIn(thread_id, stacks_to_registers_map)
            register_val_list = stacks_to_registers_map[thread_id]
            frame_register_list = frame.GetRegisters()
            # explicitly verify we collected fs and gs base for x86_64
            explicit_registers = ["fs_base", "gs_base"]
            for reg in explicit_registers:
                register = frame_register_list.GetFirstValueByName(reg)
                self.assertNotEqual(None, register)
                self.assertEqual(
                    register.GetValueAsUnsigned(),
                    stacks_to_registers_map[thread_id]
                    .GetFirstValueByName("fs_base")
                    .GetValueAsUnsigned(),
                )

            for x in register_val_list:
                self.assertEqual(
                    x.GetValueAsUnsigned(),
                    frame_register_list.GetFirstValueByName(
                        x.GetName()
                    ).GetValueAsUnsigned(),
                )

        self.dbg.DeleteTarget(target)

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_save_linux_mini_dump(self):
        """Test that we can save a Linux mini dump."""

        self.build()
        exe = self.getBuildArtifact("a.out")
        core_stack = self.getBuildArtifact("core.stack.dmp")
        core_dirty = self.getBuildArtifact("core.dirty.dmp")
        core_full = self.getBuildArtifact("core.full.dmp")
        core_sb_stack = self.getBuildArtifact("core_sb.stack.dmp")
        core_sb_dirty = self.getBuildArtifact("core_sb.dirty.dmp")
        core_sb_full = self.getBuildArtifact("core_sb.full.dmp")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)

            # get neccessary data for the verification phase
            process_info = process.GetProcessInfo()
            expected_pid = process_info.GetProcessID() if process_info.IsValid() else -1
            expected_number_of_modules = target.GetNumModules()
            expected_modules = target.modules
            expected_number_of_threads = process.GetNumThreads()
            expected_threads = []
            stacks_to_sp_map = {}
            stacks_to_registers_map = {}

            for thread_idx in range(process.GetNumThreads()):
                thread = process.GetThreadAtIndex(thread_idx)
                thread_id = thread.GetThreadID()
                expected_threads.append(thread_id)
                stacks_to_sp_map[thread_id] = thread.GetFrameAtIndex(0).GetSP()
                stacks_to_registers_map[thread_id] = thread.GetFrameAtIndex(
                    0
                ).GetRegisters()

            # save core and, kill process and verify corefile existence
            base_command = "process save-core --plugin-name=minidump "
            self.runCmd(base_command + " --style=stack '%s'" % (core_stack))
            self.assertTrue(os.path.isfile(core_stack))
            self.verify_core_file(
                core_stack,
                expected_pid,
                expected_modules,
                expected_threads,
                stacks_to_sp_map,
                stacks_to_registers_map,
            )

            self.runCmd(base_command + " --style=modified-memory '%s'" % (core_dirty))
            self.assertTrue(os.path.isfile(core_dirty))
            self.verify_core_file(
                core_dirty,
                expected_pid,
                expected_modules,
                expected_threads,
                stacks_to_sp_map,
                stacks_to_registers_map,
            )

            self.runCmd(base_command + " --style=full '%s'" % (core_full))
            self.assertTrue(os.path.isfile(core_full))
            self.verify_core_file(
                core_full,
                expected_pid,
                expected_modules,
                expected_threads,
                stacks_to_sp_map,
                stacks_to_registers_map,
            )

            options = lldb.SBSaveCoreOptions()
            core_sb_stack_spec = lldb.SBFileSpec(core_sb_stack)
            options.SetOutputFile(core_sb_stack_spec)
            options.SetPluginName("minidump")
            options.SetStyle(lldb.eSaveCoreStackOnly)
            # validate saving via SBProcess
            error = process.SaveCore(options)
            self.assertTrue(error.Success())
            self.assertTrue(os.path.isfile(core_sb_stack))
            self.verify_core_file(
                core_sb_stack,
                expected_pid,
                expected_modules,
                expected_threads,
                stacks_to_sp_map,
                stacks_to_registers_map,
            )

            options = lldb.SBSaveCoreOptions()
            core_sb_dirty_spec = lldb.SBFileSpec(core_sb_dirty)
            options.SetOutputFile(core_sb_dirty_spec)
            options.SetPluginName("minidump")
            options.SetStyle(lldb.eSaveCoreDirtyOnly)
            error = process.SaveCore(options)
            self.assertTrue(error.Success())
            self.assertTrue(os.path.isfile(core_sb_dirty))
            self.verify_core_file(
                core_sb_dirty,
                expected_pid,
                expected_modules,
                expected_threads,
                stacks_to_sp_map,
                stacks_to_registers_map,
            )

            # Minidump can now save full core files, but they will be huge and
            # they might cause this test to timeout.
            options = lldb.SBSaveCoreOptions()
            core_sb_full_spec = lldb.SBFileSpec(core_sb_full)
            options.SetOutputFile(core_sb_full_spec)
            options.SetPluginName("minidump")
            options.SetStyle(lldb.eSaveCoreFull)
            error = process.SaveCore(options)
            self.assertTrue(error.Success())
            self.assertTrue(os.path.isfile(core_sb_full))
            self.verify_core_file(
                core_sb_full,
                expected_pid,
                expected_modules,
                expected_threads,
                stacks_to_sp_map,
                stacks_to_registers_map,
            )

            self.assertSuccess(process.Kill())
        finally:
            # Clean up the mini dump file.
            self.assertTrue(self.dbg.DeleteTarget(target))
            if os.path.isfile(core_stack):
                os.unlink(core_stack)
            if os.path.isfile(core_dirty):
                os.unlink(core_dirty)
            if os.path.isfile(core_full):
                os.unlink(core_full)
            if os.path.isfile(core_sb_stack):
                os.unlink(core_sb_stack)
            if os.path.isfile(core_sb_dirty):
                os.unlink(core_sb_dirty)
            if os.path.isfile(core_sb_full):
                os.unlink(core_sb_full)

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_save_linux_mini_dump_thread_options(self):
        """Test that we can save a Linux mini dump
        with a subset of threads"""

        self.build()
        exe = self.getBuildArtifact("a.out")
        thread_subset_dmp = self.getBuildArtifact("core.thread.subset.dmp")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)

            thread_to_include = process.GetThreadAtIndex(0)
            options = lldb.SBSaveCoreOptions()
            thread_subset_spec = lldb.SBFileSpec(thread_subset_dmp)
            options.AddThread(thread_to_include)
            options.SetOutputFile(thread_subset_spec)
            options.SetPluginName("minidump")
            options.SetStyle(lldb.eSaveCoreStackOnly)
            error = process.SaveCore(options)
            self.assertTrue(error.Success())

            core_target = self.dbg.CreateTarget(None)
            core_process = core_target.LoadCore(thread_subset_dmp)

            self.assertTrue(core_process, PROCESS_IS_VALID)
            self.assertEqual(core_process.GetNumThreads(), 1)
            saved_thread = core_process.GetThreadAtIndex(0)
            expected_thread = process.GetThreadAtIndex(0)
            self.assertEqual(expected_thread.GetThreadID(), saved_thread.GetThreadID())
            expected_sp = expected_thread.GetFrameAtIndex(0).GetSP()
            saved_sp = saved_thread.GetFrameAtIndex(0).GetSP()
            self.assertEqual(expected_sp, saved_sp)
            expected_region = lldb.SBMemoryRegionInfo()
            saved_region = lldb.SBMemoryRegionInfo()
            error = core_process.GetMemoryRegionInfo(saved_sp, saved_region)
            self.assertTrue(error.Success(), error.GetCString())
            error = process.GetMemoryRegionInfo(expected_sp, expected_region)
            self.assertTrue(error.Success(), error.GetCString())
            self.assertEqual(
                expected_region.GetRegionBase(), saved_region.GetRegionBase()
            )
            self.assertEqual(
                expected_region.GetRegionEnd(), saved_region.GetRegionEnd()
            )

        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))
            if os.path.isfile(thread_subset_dmp):
                os.unlink(thread_subset_dmp)

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_save_linux_mini_dump_default_options(self):
        """Test that we can save a Linux mini dump with default SBSaveCoreOptions"""

        self.build()
        exe = self.getBuildArtifact("a.out")
        default_value_file = self.getBuildArtifact("core.defaults.dmp")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)

            process_info = process.GetProcessInfo()
            expected_pid = process_info.GetProcessID() if process_info.IsValid() else -1
            expected_modules = target.modules
            expected_threads = []
            stacks_to_sp_map = {}
            expected_pid = process.GetProcessInfo().GetProcessID()
            stacks_to_registers_map = {}

            for thread_idx in range(process.GetNumThreads()):
                thread = process.GetThreadAtIndex(thread_idx)
                thread_id = thread.GetThreadID()
                expected_threads.append(thread_id)
                stacks_to_sp_map[thread_id] = thread.GetFrameAtIndex(0).GetSP()
                stacks_to_registers_map[thread_id] = thread.GetFrameAtIndex(
                    0
                ).GetRegisters()

            # This is almost identical to the single thread test case because
            # minidump defaults to stacks only, so we want to see if the
            # default options work as expected.
            options = lldb.SBSaveCoreOptions()
            default_value_spec = lldb.SBFileSpec(default_value_file)
            options.SetOutputFile(default_value_spec)
            options.SetPluginName("minidump")
            error = process.SaveCore(options)
            self.assertTrue(error.Success())

            self.verify_core_file(
                default_value_file,
                expected_pid,
                expected_modules,
                expected_threads,
                stacks_to_sp_map,
                stacks_to_registers_map,
            )

        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))
            if os.path.isfile(default_value_file):
                os.unlink(default_value_file)

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_save_linux_minidump_one_region(self):
        """Test that we can save a Linux mini dump with one region in sbsavecore regions"""

        self.build()
        exe = self.getBuildArtifact("a.out")
        one_region_file = self.getBuildArtifact("core.one_region.dmp")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)

            memory_region = lldb.SBMemoryRegionInfo()
            memory_list = process.GetMemoryRegions()
            memory_list.GetMemoryRegionAtIndex(0, memory_region)

            # This is almost identical to the single thread test case because
            # minidump defaults to stacks only, so we want to see if the
            # default options work as expected.
            options = lldb.SBSaveCoreOptions()
            file_spec = lldb.SBFileSpec(one_region_file)
            options.SetOutputFile(file_spec)
            options.SetPluginName("minidump")
            options.AddMemoryRegionToSave(memory_region)
            options.SetStyle(lldb.eSaveCoreCustomOnly)
            error = process.SaveCore(options)
            print(f"Error: {error.GetCString()}")
            self.assertTrue(error.Success(), error.GetCString())

            core_target = self.dbg.CreateTarget(None)
            core_proc = core_target.LoadCore(one_region_file)
            core_memory_list = core_proc.GetMemoryRegions()
            # Note because the /proc/pid maps are included on linux, we can't
            # depend on size for validation, so we'll ensure the first region
            # is present and then assert we fail on the second.
            core_memory_region = lldb.SBMemoryRegionInfo()
            core_memory_list.GetMemoryRegionAtIndex(0, core_memory_region)
            self.assertEqual(
                core_memory_region.GetRegionBase(), memory_region.GetRegionBase()
            )
            self.assertEqual(
                core_memory_region.GetRegionEnd(), memory_region.GetRegionEnd()
            )

            region_two = lldb.SBMemoryRegionInfo()
            core_memory_list.GetMemoryRegionAtIndex(1, region_two)
            err = lldb.SBError()
            content = core_proc.ReadMemory(region_two.GetRegionBase(), 1, err)
            self.assertTrue(err.Fail(), "Should fail to read memory")

        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))
            if os.path.isfile(one_region_file):
                os.unlink(one_region_file)

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_save_minidump_custom_save_style(self):
        """Test that verifies a custom and unspecified save style fails for
        containing no data to save"""

        self.build()
        exe = self.getBuildArtifact("a.out")
        custom_file = self.getBuildArtifact("core.custom.dmp")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)

            options = lldb.SBSaveCoreOptions()
            options.SetOutputFile(lldb.SBFileSpec(custom_file))
            options.SetPluginName("minidump")
            options.SetStyle(lldb.eSaveCoreCustomOnly)

            error = process.SaveCore(options)
            self.assertTrue(error.Fail())
            self.assertEqual(
                error.GetCString(), "no valid address ranges found for core style"
            )

        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))
            if os.path.isfile(custom_file):
                os.unlink(custom_file)

    def save_core_with_region(self, process, region_index):
        try:
            custom_file = self.getBuildArtifact("core.custom.dmp")
            memory_region = lldb.SBMemoryRegionInfo()
            memory_list = process.GetMemoryRegions()
            memory_list.GetMemoryRegionAtIndex(0, memory_region)
            options = lldb.SBSaveCoreOptions()
            options.SetOutputFile(lldb.SBFileSpec(custom_file))
            options.SetPluginName("minidump")
            options.SetStyle(lldb.eSaveCoreFull)

            error = process.SaveCore(options)
            self.assertTrue(error.Success())
            core_target = self.dbg.CreateTarget(None)
            core_proc = core_target.LoadCore(custom_file)
            core_memory_list = core_proc.GetMemoryRegions()
            # proc/pid/ maps are included on linux, so we can't depend on size
            # for validation, we make a set of all the ranges,
            # and ensure no duplicates!
            range_set = set()
            for x in range(core_memory_list.GetSize()):
                core_memory_region = lldb.SBMemoryRegionInfo()
                core_memory_list.GetMemoryRegionAtIndex(x, core_memory_region)
                mem_tuple = (
                    core_memory_region.GetRegionBase(),
                    core_memory_region.GetRegionEnd(),
                )
                self.assertTrue(
                    mem_tuple not in range_set, "Duplicate memory region found"
                )
                range_set.add(mem_tuple)
        finally:
            if os.path.isfile(custom_file):
                os.unlink(custom_file)

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_save_minidump_custom_save_style_duplicated_regions(self):
        """Test that verifies a custom and unspecified save style fails for
        containing no data to save"""

        self.build()
        exe = self.getBuildArtifact("a.out")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)

            memory_list = process.GetMemoryRegions()
            # Test that we don't duplicate regions, by duplicating regions
            # at various indices.
            self.save_core_with_region(process, 0)
            self.save_core_with_region(process, len(memory_list) - 1)

        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))

    @skipUnlessPlatform(["linux"])
    def minidump_deleted_on_save_failure(self):
        """Test that verifies the minidump file is deleted after an error"""

        self.build()
        exe = self.getBuildArtifact("a.out")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)

            custom_file = self.getBuildArtifact("core.should.be.deleted.custom.dmp")
            options = lldb.SBSaveCoreOptions()
            options.SetOutputFile(lldb.SBFileSpec(custom_file))
            options.SetPluginName("minidump")
            options.SetStyle(lldb.eSaveCoreCustomOnly)
            # We set custom only and have no thread list and have no memory.
            error = process.SaveCore(options)
            self.assertTrue(error.Fail())
            self.assertIn(
                "no valid address ranges found for core style", error.GetCString()
            )
            self.assertTrue(not os.path.isfile(custom_file))

        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))

    @skipUnlessPlatform(["linux"])
    @skipUnlessArch("x86_64")
    def minidump_saves_fs_base_region(self):
        """Test that verifies the minidump file saves region for fs_base"""

        self.build()
        exe = self.getBuildArtifact("a.out")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)
            thread = process.GetThreadAtIndex(0)
            custom_file = self.getBuildArtifact("core.reg_region.dmp")
            options = lldb.SBSaveCoreOptions()
            options.SetOutputFile(lldb.SBFileSpec(custom_file))
            options.SetPluginName("minidump")
            options.SetStyle(lldb.eSaveCoreCustomOnly)
            options.AddThread(thread)
            error = process.SaveCore(options)
            self.assertTrue(error.Success())

            registers = thread.GetFrameAtIndex(0).GetRegisters()
            fs_base = registers.GetFirstValueByName("fs_base").GetValueAsUnsigned()
            self.assertTrue(fs_base != 0)
            core_target = self.dbg.CreateTarget(None)
            core_proc = core_target.LoadCore(one_region_file)
            core_region_list = core_proc.GetMemoryRegions()
            live_region_list = process.GetMemoryRegions()
            live_region = lldb.SBMemoryRegionInfo()
            live_region_list.GetMemoryRegionForAddress(fs_base, live_region)
            core_region = lldb.SBMemoryRegionInfo()
            error = core_region_list.GetMemoryRegionForAddress(fs_base, core_region)
            self.assertTrue(error.Success())
            self.assertEqual(live_region, core_region)

        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))
            self.assertTrue(self.dbg.DeleteTarget(core_target))
            if os.path.isfile(custom_file):
                os.unlink(custom_file)

    def minidump_deterministic_difference(self):
        """Test that verifies that two minidumps produced are identical."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )

            core_styles = [
                lldb.eSaveCoreStackOnly,
                lldb.eSaveCoreDirtyOnly,
                lldb.eSaveCoreFull,
            ]
            for style in core_styles:
                spec_one = lldb.SBFileSpec(self.getBuildArtifact("core.one.dmp"))
                spec_two = lldb.SBFileSpec(self.getBuildArtifact("core.two.dmp"))
                options = lldb.SBSaveCoreOptions()
                options.SetOutputFile(spec_one)
                options.SetPluginName("minidump")
                options.SetStyle(style)
                error = process.SaveCore(options)
                self.assertTrue(error.Success())
                options.SetOutputFile(spec_two)
                error = process.SaveCore(options)
                self.assertTrue(error.Success())

                file_one = None
                file_two = None
                with open(spec_one.GetFileName(), mode="rb") as file:
                    file_one = file.read()
                with open(spec_two.GetFileName(), mode="rb") as file:
                    file_two = file.read()
                self.assertEqual(file_one, file_two)
                self.assertTrue(os.unlink(spec_one.GetFileName()))
                self.assertTrue(os.unlink(spec_two.GetFileName()))
        finally:
            self.assertTrue(self.dbg.DeleteTarget(target))

    @skipUnlessPlatform(["linux"])
    @skipUnlessArch("x86_64")
    def minidump_saves_fs_base_region(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        try:
            target = self.dbg.CreateTarget(exe)
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)
            thread = process.GetThreadAtIndex(0)
            tls_file = self.getBuildArtifact("core.tls.dmp")
            options = lldb.SBSaveCoreOptions()
            options.SetOutputFile(lldb.SBFileSpec(tls_file))
            options.SetPluginName("minidump")
            options.SetStyle(lldb.eSaveCoreCustomOnly)
            options.AddThread(thread)
            error = process.SaveCore(options)
            self.assertTrue(error.Success())
            core_target = self.dbg.CreateTarget(None)
            core_proc = core_target.LoadCore(tls_file)
            frame = core_proc.GetThreadAtIndex(0).GetFrameAtIndex(0)
            tls_val = frame.FindValue("lf")
            self.assertEqual(tls_val.GetValueAsUnsigned(), 42)

        except:
            self.assertTrue(self.dbg.DeleteTarget(target))
            if os.path.isfile(tls_file):
                os.unlink(tls_file)
