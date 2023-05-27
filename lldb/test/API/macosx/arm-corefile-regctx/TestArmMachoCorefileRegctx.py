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

    @skipUnlessDarwin
    def setUp(self):
        TestBase.setUp(self)
        self.build()
        self.create_corefile = self.getBuildArtifact("a.out")
        self.corefile = self.getBuildArtifact("core")

    def test_armv7_corefile(self):
        ### Create corefile
        retcode = call(self.create_corefile + " armv7 " + self.corefile, shell=True)

        target = self.dbg.CreateTarget("")
        err = lldb.SBError()
        process = target.LoadCore(self.corefile)
        self.assertEqual(process.IsValid(), True)
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

    def test_arm64_corefile(self):
        ### Create corefile
        retcode = call(self.create_corefile + " arm64 " + self.corefile, shell=True)

        target = self.dbg.CreateTarget("")
        err = lldb.SBError()
        process = target.LoadCore(self.corefile)
        self.assertEqual(process.IsValid(), True)
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
