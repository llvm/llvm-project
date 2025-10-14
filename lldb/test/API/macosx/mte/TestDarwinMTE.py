"""Test MTE Memory Tagging on Apple platforms"""

import lldb
import re
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbsuite.test.cpu_feature as cpu_feature

exe_name = "uaf_mte"  # Must match Makefile


class TestDarwinMTE(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessFeature(cpu_feature.AArch64.MTE)
    def test_tag_fault(self):
        self.build()
        exe = self.getBuildArtifact(exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        process = target.LaunchSimple(None, None, None)
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        self.expect(
            "thread info",
            substrs=[
                "stop reason = EXC_ARM_MTE_TAG_FAULT",
                "MTE tag mismatch detected",
            ],
        )

    @skipUnlessFeature(cpu_feature.AArch64.MTE)
    def test_memory_region(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// before free", lldb.SBFileSpec("main.c"), exe_name=exe_name
        )

        # (lldb) memory region ptr
        # [0x00000001005ec000-0x00000001009ec000) rw-
        # memory tagging: enabled
        # Modified memory (dirty) page list provided, 2 entries.
        # Dirty pages: 0x1005ec000, 0x1005fc000.
        self.expect("memory region ptr", substrs=["memory tagging: enabled"])

    @skipUnlessFeature(cpu_feature.AArch64.MTE)
    def test_memory_read_show_tags(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// before free", lldb.SBFileSpec("main.c"), exe_name=exe_name
        )

        # (lldb) memory read ptr-16 ptr+48 --show-tags
        # 0x7d2c00930: 00 00 00 00 00 00 00 00 d0 e3 a5 0a 02 00 00 00  ................ (tag: 0x3)
        # 0x7d2c00940: 48 65 6c 6c 6f 00 00 00 00 00 00 00 00 00 00 00  Hello........... (tag: 0xb)
        # 0x7d2c00950: 57 6f 72 6c 64 00 00 00 00 00 00 00 00 00 00 00  World........... (tag: 0xb)
        # 0x7d2c00960: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................ (tag: 0x9)
        self.expect(
            "memory read ptr-16 ptr+48 --show-tags",
            substrs=[" Hello...........", " World..........."],
            patterns=[r"(.*\(tag: 0x[0-9a-f]\)\n){4}"],
        )

    def _parse_pointer_tag(self, output):
        return re.search(r"Logical tag: (0x[0-9a-f])", output).group(1)

    def _parse_memory_tags(self, output, expected_tag_count):
        tags = re.findall(r"\): (0x[0-9a-f])", output)
        self.assertEqual(len(tags), expected_tag_count)
        return tags

    @skipUnlessFeature(cpu_feature.AArch64.MTE)
    def test_memory_tag_read(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// before free", lldb.SBFileSpec("main.c"), exe_name=exe_name
        )

        # (lldb) memory tag read ptr-1 ptr+33
        # Logical tag: 0x5
        # Allocation tags:
        # [0x100a65a40, 0x100a65a50): 0xf (mismatch)
        # [0x100a65a50, 0x100a65a60): 0x5
        # [0x100a65a60, 0x100a65a70): 0x5
        # [0x100a65a70, 0x100a65a80): 0x2 (mismatch)
        self.expect(
            "memory tag read ptr-1 ptr+33",
            substrs=["Logical tag: 0x", "Allocation tags:", "(mismatch)"],
            patterns=[r"(\[.*\): 0x[0-9a-f].*\n){4}"],
        )
        output = self.res.GetOutput()
        self.assertEqual(output.count("(mismatch)"), 2)
        ptr_tag = self._parse_pointer_tag(output)
        tags = self._parse_memory_tags(output, 4)
        self.assertEqual(tags[1], ptr_tag)
        self.assertEqual(tags[2], ptr_tag)
        self.assertNotEqual(tags[0], ptr_tag)  # Memory that comes before/after
        self.assertNotEqual(tags[3], ptr_tag)  # allocation has different tag.

        # Continue running until MTE fault
        self.expect("process continue", substrs=["stop reason = EXC_ARM_MTE_TAG_FAULT"])

        self.runCmd("memory tag read ptr-1 ptr+33")
        output = self.res.GetOutput()
        self.assertEqual(output.count("(mismatch)"), 4)
        tags = self._parse_memory_tags(output, 4)
        self.assertTrue(all(t != ptr_tag for t in tags))
