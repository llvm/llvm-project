"""
Test saving a minidump when several threads have their stack pointers in the
same memory region: every thread must be in the ThreadList exactly once.
"""

import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

THREAD_LIST_STREAM = 3  # llvm::minidump::StreamType::ThreadList
# llvm::minidump::Thread: ThreadId, SuspendCount, PriorityClass, Priority,
# EnvironmentBlock, Stack (StartOfMemoryRange, DataSize, RVA),
# Context (DataSize, RVA).
THREAD_FORMAT = "<IIIIQQIIII"


class ProcessSaveCoreMinidumpSharedStackRegionTestCase(TestBase):
    def read_thread_list(self, core_path):
        """Return the raw llvm::minidump::Thread entries of the ThreadList."""
        with open(core_path, "rb") as core_file:
            core_bytes = core_file.read()
        self.assertEqual(core_bytes[:4], b"MDMP")
        num_streams, directory_rva = struct.unpack_from("<II", core_bytes, 8)
        for stream_index in range(num_streams):
            stream_type, _, stream_rva = struct.unpack_from(
                "<III", core_bytes, directory_rva + stream_index * 12
            )
            if stream_type != THREAD_LIST_STREAM:
                continue
            (num_threads,) = struct.unpack_from("<I", core_bytes, stream_rva)
            thread_size = struct.calcsize(THREAD_FORMAT)
            return [
                struct.unpack_from(
                    THREAD_FORMAT, core_bytes, stream_rva + 4 + i * thread_size
                )
                for i in range(num_threads)
            ]
        self.fail("minidump has no ThreadList stream")

    @requireMinidumpSave
    @skipUnlessPlatform(["linux"])
    def test_threads_sharing_stack_region(self):
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        frame = thread.GetFrameAtIndex(0)
        num_workers = frame.FindVariable("num_threads").GetValueAsUnsigned()
        stacks = frame.FindVariable("stacks").GetValueAsUnsigned()
        stack_size = frame.FindVariable("stack_size").GetValueAsUnsigned()
        stacks_end = stacks + num_workers * stack_size

        live_tids = []
        live_sps = {}
        for live_thread in process:
            tid = live_thread.GetThreadID()
            live_tids.append(tid)
            live_sps[tid] = live_thread.GetFrameAtIndex(0).GetSP()

        # Make sure the workers really share one memory region, otherwise this
        # test doesn't test anything.
        region = lldb.SBMemoryRegionInfo()
        self.assertSuccess(process.GetMemoryRegionInfo(stacks, region))
        worker_tids = [tid for tid, sp in live_sps.items() if stacks <= sp < stacks_end]
        self.assertEqual(len(worker_tids), num_workers)
        for tid in worker_tids:
            self.assertGreaterEqual(live_sps[tid], region.GetRegionBase())
            self.assertLess(live_sps[tid], region.GetRegionEnd())

        core_path = self.getBuildArtifact("shared_stack_region.dmp")
        options = lldb.SBSaveCoreOptions()
        options.SetOutputFile(lldb.SBFileSpec(core_path))
        options.SetPluginName("minidump")
        options.SetStyle(lldb.eSaveCoreStackOnly)
        self.assertSuccess(process.SaveCore(options))

        # The ThreadList has every thread once, in the live process' order,
        # each with its own context and a stack descriptor covering its SP.
        thread_list = self.read_thread_list(core_path)
        self.assertEqual([entry[0] for entry in thread_list], live_tids)
        context_rvas = [entry[9] for entry in thread_list]
        self.assertEqual(len(set(context_rvas)), len(context_rvas))
        for entry in thread_list:
            tid, stack_start, stack_size = entry[0], entry[5], entry[6]
            self.assertGreater(stack_size, 0, f"empty stack for thread {tid}")
            self.assertGreaterEqual(live_sps[tid], stack_start)
            self.assertLess(live_sps[tid], stack_start + stack_size)

        core_target = self.dbg.CreateTarget(None)
        core_process = core_target.LoadCore(core_path)
        self.assertTrue(core_process, PROCESS_IS_VALID)
        self.assertEqual(
            [core_thread.GetThreadID() for core_thread in core_process], live_tids
        )
        for core_thread in core_process:
            sp = core_thread.GetFrameAtIndex(0).GetSP()
            self.assertEqual(sp, live_sps[core_thread.GetThreadID()])
            error = lldb.SBError()
            core_process.ReadMemory(sp, 1, error)
            self.assertSuccess(error)
        self.dbg.DeleteTarget(core_target)
