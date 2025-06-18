"""Test that lldb will read additional registers from Mach-O LC_NOTE metadata."""

import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMetadataRegisters(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIfRemote
    def test_add_registers_via_metadata(self):
        self.build()
        self.aout_exe = self.getBuildArtifact("a.out")
        lldb_corefile = self.getBuildArtifact("lldb.core")
        metadata_corefile = self.getBuildArtifact("metadata.core")
        add_lcnote = self.getBuildArtifact("add-lcnote")

        (target, process, t, bp) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        self.assertTrue(process.IsValid())

        if self.TraceOn():
            self.runCmd("bt")
            self.runCmd("reg read -a")

        self.runCmd("process save-core " + lldb_corefile)
        process.Kill()
        target.Clear()

        cmd = (
            add_lcnote
            + " "
            + "-r"
            + " "
            + "-i '%s'" % lldb_corefile
            + " "
            + "-o '%s'" % metadata_corefile
            + " "
            + "-n 'process metadata' "
            + "-s '"
            + """{"threads":[{"register_info":
                   {"sets":["Special Registers", "General Purpose Registers"],
                    "registers":[
                       {"name":"jar", "value":10, "bitsize": 32, "set": 0},
                       {"name":"bar", "value":65537, "bitsize":16, "set":0},
                       {"name":"mar", "value":65537, "bitsize":32, "set":0},
                       {"name":"anotherpc", "value":55, "bitsize":64, "set": 1}]}}]}"""
            + "'"
        )

        print(cmd)
        call(cmd, shell=True)

        # Now load the corefile
        target = self.dbg.CreateTarget("")
        process = target.LoadCore(metadata_corefile)
        self.assertTrue(process.IsValid())
        if self.TraceOn():
            self.runCmd("bt")
            self.runCmd("reg read -a")

        thread = process.GetSelectedThread()
        frame = thread.GetFrameAtIndex(0)

        # Register sets will be
        #    from LC_THREAD:
        #       General Purpose Registers
        #       Floating Point Registers
        #       Exception State Registers
        #    from LC_NOTE metadata:
        #       Special Registers
        self.assertEqual(frame.registers[0].GetName(), "General Purpose Registers")
        self.assertEqual(frame.registers[3].GetName(), "Special Registers")

        anotherpc = frame.registers[0].GetChildMemberWithName("anotherpc")
        self.assertTrue(anotherpc.IsValid())
        self.assertEqual(anotherpc.GetValueAsUnsigned(), 0x37)

        jar = frame.registers[3].GetChildMemberWithName("jar")
        self.assertTrue(jar.IsValid())
        self.assertEqual(jar.GetValueAsUnsigned(), 10)
        self.assertEqual(jar.GetByteSize(), 4)

        bar = frame.registers[3].GetChildMemberWithName("bar")
        self.assertTrue(bar.IsValid())
        self.assertEqual(bar.GetByteSize(), 2)

        mar = frame.registers[3].GetChildMemberWithName("mar")
        self.assertTrue(mar.IsValid())
        self.assertEqual(mar.GetValueAsUnsigned(), 0x10001)
        self.assertEqual(mar.GetByteSize(), 4)
