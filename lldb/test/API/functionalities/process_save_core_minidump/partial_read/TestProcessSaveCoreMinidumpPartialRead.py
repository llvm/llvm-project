"""
Test saving a minidump when a saved memory range contains an unreadable page:
the readable memory after the hole must still be captured.
"""

import os
import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

MEMORY64_LIST_STREAM = 9  # llvm::minidump::StreamType::Memory64List


class ProcessSaveCoreMinidumpPartialReadTestCase(TestBase):
    def assert_memory64_datasize_within_file(self, core_path):
        """The Memory64List must not claim more bytes than the file holds."""
        with open(core_path, "rb") as core_file:
            core_bytes = core_file.read()
        self.assertEqual(core_bytes[:4], b"MDMP")
        num_streams, directory_rva = struct.unpack_from("<II", core_bytes, 8)

        memory64_list_found = False
        for stream_index in range(num_streams):
            stream_type, _, stream_rva = struct.unpack_from(
                "<III", core_bytes, directory_rva + stream_index * 12
            )
            if stream_type != MEMORY64_LIST_STREAM:
                continue
            memory64_list_found = True
            num_ranges, base_rva = struct.unpack_from("<QQ", core_bytes, stream_rva)
            total_data_size = sum(
                struct.unpack_from("<QQ", core_bytes, stream_rva + 16 + i * 16)[1]
                for i in range(num_ranges)
            )
            self.assertLessEqual(
                base_rva + total_data_size,
                len(core_bytes),
                "Memory64List DataSize claims more bytes than the minidump holds",
            )
        self.assertTrue(memory64_list_found, "minidump has no Memory64List stream")

    @skipUnlessPlatform(["linux"])
    def test_save_core_range_with_unreadable_tail(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        lldbutil.run_break_set_by_source_regexp(self, "Set a breakpoint here")
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetFrameAtIndex(0)
        region = frame.FindVariable("region").GetValueAsUnsigned()
        page = frame.FindVariable("page").GetValueAsUnsigned()
        self.assertNotEqual(region, 0)
        self.assertNotEqual(page, 0)

        # The first page reads, the tail past the file's end does not.
        live_error = lldb.SBError()
        live_page = process.ReadMemory(region, page, live_error)
        self.assertSuccess(live_error)
        self.assertEqual(live_page, b"\xab" * page)
        tail_error = lldb.SBError()
        process.ReadMemory(region + page, page, tail_error)
        self.assertTrue(tail_error.Fail())

        core_path = self.getBuildArtifact("partial_read.dmp")
        options = lldb.SBSaveCoreOptions()
        options.SetOutputFile(lldb.SBFileSpec(core_path))
        options.SetPluginName("minidump")
        options.SetStyle(lldb.eSaveCoreCustomOnly)
        rw = 0b110  # ePermissionsReadable | ePermissionsWritable
        options.AddMemoryRegionToSave(
            lldb.SBMemoryRegionInfo("", region, region + 4 * page, rw, True)
        )
        self.assertSuccess(process.SaveCore(options))

        core_target = None
        try:
            core_target = self.dbg.CreateTarget(None)
            core_process = core_target.LoadCore(core_path)
            self.assertTrue(core_process.IsValid())

            core_error = lldb.SBError()
            core_page = core_process.ReadMemory(region, page, core_error)
            self.assertSuccess(core_error)
            self.assertEqual(
                core_page,
                b"\xab" * page,
                "readable page before the unreadable tail was lost or misaligned",
            )

            self.assert_memory64_datasize_within_file(core_path)
        finally:
            self.dbg.DeleteTarget(target)
            if core_target:
                self.dbg.DeleteTarget(core_target)
            if os.path.isfile(core_path):
                os.unlink(core_path)
