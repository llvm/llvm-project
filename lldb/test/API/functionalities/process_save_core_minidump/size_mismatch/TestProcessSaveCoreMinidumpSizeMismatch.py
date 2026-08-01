"""
Regression test for the Memory64List size accounting in minidump save-core.

When a saved range contains an unreadable page the read fails part-way. The
range's DataSize must equal the number of bytes actually written to the shared
Memory64 blob; if it instead records the number of bytes ReadMemoryInChunks read
(which includes the partially-read bytes that were dropped on the error), the
blob's cumulative offsets desync and the descriptors claim more data than the
file holds. See MinidumpFileBuilder::ReadWriteMemoryInChunks.
"""

import os
import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

MEMORY64_LIST_STREAM = 9  # llvm::minidump::StreamType::Memory64List


class ProcessSaveCoreMinidumpSizeMismatchTestCase(TestBase):
    def assert_memory64_datasize_within_file(self, core_path):
        """Assert the minidump has a Memory64List whose ranges' cumulative
        DataSize does not claim more bytes than the file actually holds."""
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
    def test_memory64_datasize_matches_written_bytes(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        lldbutil.run_break_set_by_source_regexp(self, "Set a breakpoint here")
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped)

        frame = process.GetSelectedThread().GetFrameAtIndex(0)
        readable_region = frame.FindVariable("readable_region").GetValueAsUnsigned()
        readable_region_size = frame.FindVariable(
            "readable_region_size"
        ).GetValueAsUnsigned()
        unreadable_tail_region = frame.FindVariable(
            "unreadable_tail_region"
        ).GetValueAsUnsigned()
        page = frame.FindVariable("page").GetValueAsUnsigned()
        self.assertNotEqual(readable_region, 0)
        self.assertNotEqual(unreadable_tail_region, 0)

        core_path = self.getBuildArtifact("size_mismatch.dmp")
        options = lldb.SBSaveCoreOptions()
        options.SetOutputFile(lldb.SBFileSpec(core_path))
        options.SetPluginName("minidump")
        options.SetStyle(lldb.eSaveCoreCustomOnly)
        rw = 0b110  # ePermissionsReadable | ePermissionsWritable
        options.AddMemoryRegionToSave(
            lldb.SBMemoryRegionInfo(
                "", readable_region, readable_region + readable_region_size, rw, True
            )
        )
        options.AddMemoryRegionToSave(
            lldb.SBMemoryRegionInfo(
                "", unreadable_tail_region, unreadable_tail_region + 4 * page, rw, True
            )
        )
        self.assertSuccess(process.SaveCore(options))

        try:
            # The range with the unreadable tail must not make the Memory64List
            # claim more data than the file actually contains.
            self.assert_memory64_datasize_within_file(core_path)

            # The fully-readable range must round-trip byte-for-byte (it must not
            # be shifted by a preceding range's over-claimed DataSize).
            core_target = self.dbg.CreateTarget(None)
            core_process = core_target.LoadCore(core_path)
            self.assertTrue(core_process.IsValid())
            err = lldb.SBError()
            got = core_process.ReadMemory(readable_region, page, err)
            self.assertSuccess(err)
            self.assertEqual(got, b"\xcd" * page)
        finally:
            if os.path.isfile(core_path):
                os.unlink(core_path)
