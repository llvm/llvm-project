"""Test that Mach-O armv7/arm64 corefile register contexts are read by lldb."""

import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestArmMachoCorefileRegctx(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin # CI fail on Windows, lr has value 0x0F000000?
    def test_armv7_corefile(self):
        ### Create corefile
        corefile = self.getBuildArtifact("core")
        self.yaml2macho_core("armv7m.yaml", corefile)

        target = self.dbg.CreateTarget("")
        err = lldb.SBError()
        process = target.LoadCore(corefile)
        self.assertTrue(process.IsValid())
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        lr = frame.FindRegister("lr")
        self.assertTrue(lr.IsValid())
        self.assertEqual(lr.GetValueAsUnsigned(), 0x000F0000)

        pc = frame.FindRegister("pc")
        self.assertTrue(pc.IsValid())
        self.assertEqual(pc.GetValueAsUnsigned(), 0x00100000)

        exception = frame.FindRegister("exception")
        self.assertTrue(exception.IsValid())
        self.assertEqual(exception.GetValueAsUnsigned(), 0x00003F5C)

        # read 4 bytes starting at $sp-1 (an odd/unaligned address on this arch),
        # formatted hex.
        # aka `mem read -f x -s 1 -c 4 $sp-1`
        self.expect("x/4bx $sp-1", substrs=["0x000dffff", "0x1f 0x20 0x21 0x22"])

    def test_arm64_corefile(self):
        ### Create corefile
        corefile = self.getBuildArtifact("core")
        self.yaml2macho_core("arm64.yaml", corefile)

        target = self.dbg.CreateTarget("")
        err = lldb.SBError()
        process = target.LoadCore(corefile)
        self.assertTrue(process.IsValid())
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        lr = frame.FindRegister("lr")
        self.assertTrue(lr.IsValid())
        self.assertEqual(lr.GetValueAsUnsigned(), 0x000000018CD97F28)

        pc = frame.FindRegister("pc")
        self.assertTrue(pc.IsValid())
        self.assertEqual(pc.GetValueAsUnsigned(), 0x0000000100003F5C)

        exception = frame.FindRegister("far")
        self.assertTrue(exception.IsValid())
        self.assertEqual(exception.GetValueAsUnsigned(), 0x0000000100003F5C)
