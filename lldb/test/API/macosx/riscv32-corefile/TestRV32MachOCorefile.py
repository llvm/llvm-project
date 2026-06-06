"""Test that all of the GPR registers are read correctly from a riscv32 corefile."""

import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestRV32MachOCorefile(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @no_debug_info_test
    def test_riscv32_gpr_corefile_registers(self):
        corefile = self.getBuildArtifact("core")
        self.yaml2macho_core("riscv32-registers.yaml", corefile)

        target = self.dbg.CreateTarget("")
        process = target.LoadCore(corefile)

        process = target.GetProcess()
        self.assertEqual(process.GetNumThreads(), 2)

        thread = process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetNumFrames(), 1)

        frame = thread.GetFrameAtIndex(0)
        gpr_regs = frame.registers.GetValueAtIndex(0)

        self.assertEqual(gpr_regs.GetName(), "General Purpose Registers")
        self.assertEqual(gpr_regs.GetNumChildren(), 33)
        regnames = [
            "zero",
            "ra",
            "sp",
            "gp",
            "tp",
            "t0",
            "t1",
            "t2",
            "fp",
            "s1",
            "a0",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
            "s7",
            "s8",
            "s9",
            "s10",
            "s11",
            "t3",
            "t4",
            "t5",
            "t6",
            "pc",
        ]

        idx = 0
        while idx < len(regnames):
            self.assertEqual(gpr_regs.GetChildAtIndex(idx).GetName(), regnames[idx])
            idx = idx + 1

        idx = 0
        while idx < len(regnames):
            val = idx | (idx << 8) | (idx << 16) | (idx << 24)
            self.assertEqual(gpr_regs.GetChildAtIndex(idx).GetValueAsUnsigned(), val)
            idx = idx + 1

        thread = process.GetThreadAtIndex(1)
        self.assertEqual(thread.GetNumFrames(), 1)

        frame = thread.GetFrameAtIndex(0)
        gpr_regs = frame.registers.GetValueAtIndex(0)

        self.assertEqual(gpr_regs.GetChildAtIndex(0).GetValueAsUnsigned(), 0x90000000)
        self.assertEqual(gpr_regs.GetChildAtIndex(32).GetValueAsUnsigned(), 0x90202020)
