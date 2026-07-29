"""
Regression test for getModuleFileSize using sect_sp instead of next_sect_sp,
producing incorrect SizeOfImage when contiguous sections have different sizes.
"""

import os
import struct
import subprocess
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class MinidumpSizeOfImageTestCase(TestBase):
    def build_shared_lib(self):
        """Build a shared library with two contiguous PT_LOAD segments
        of different sizes via a linker script."""
        testlib_src = os.path.join(self.getSourceDir(), "testlib.c")
        testlib_out = self.getBuildArtifact("libtestlib.so")
        lds = self.getBuildArtifact("contig.lds")

        # Pad the RO segment to a page boundary so the RW segment starts
        # immediately after, creating contiguous segments of different sizes.
        with open(lds, "w") as f:
            f.write(
                "PHDRS { ro PT_LOAD FLAGS(5); rw PT_LOAD FLAGS(6);"
                " dyn PT_DYNAMIC FLAGS(6); }\n"
                "SECTIONS {\n"
                "  . = SIZEOF_HEADERS;\n"
                "  .hash : { *(.hash) } :ro\n"
                "  .gnu.hash : { *(.gnu.hash) } :ro\n"
                "  .dynsym : { *(.dynsym) } :ro\n"
                "  .dynstr : { *(.dynstr) } :ro\n"
                "  .rela.dyn : { *(.rela.dyn) } :ro\n"
                "  .rela.plt : { *(.rela.plt) } :ro\n"
                "  .plt : { *(.plt) } :ro\n"
                "  .text : { *(.text .text.*) } :ro\n"
                "  .rodata : { *(.rodata .rodata.*) } :ro\n"
                "  .eh_frame_hdr : { *(.eh_frame_hdr) } :ro\n"
                "  .eh_frame : { *(.eh_frame) . = ALIGN(0x1000); } :ro\n"
                "  .dynamic : { *(.dynamic) } :rw :dyn\n"
                "  .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) } :rw\n"
                "  .got : { *(.got) } :rw\n"
                "  .got.plt : { *(.got.plt) } :rw\n"
                "  .data : { *(.data .data.*) } :rw\n"
                "  .bss : { *(.bss .bss.*) } :rw\n"
                "  /DISCARD/ : { *(.comment) *(.note.*)"
                " *(.gnu.build.attributes) }\n"
                "}\n"
            )

        subprocess.check_call(
            [
                self.getCompiler(),
                "-shared",
                "-fPIC",
                "-Wl,-T," + lds,
                "-o",
                testlib_out,
                testlib_src,
            ]
        )
        return testlib_out

    def get_minidump_size_of_image(self, filepath, module_name):
        """Return SizeOfImage from a minidump for the module whose name
        contains module_name."""
        with open(filepath, "rb") as f:
            data = f.read()

        num_streams = struct.unpack_from("<I", data, 8)[0]
        stream_dir_rva = struct.unpack_from("<I", data, 12)[0]

        for i in range(num_streams):
            stype, _, rva = struct.unpack_from("<III", data, stream_dir_rva + i * 12)
            if stype != 4:  # ModuleListStream
                continue
            num_modules = struct.unpack_from("<I", data, rva)[0]
            for j in range(num_modules):
                off = rva + 4 + j * 108
                size_of_image = struct.unpack_from("<I", data, off + 8)[0]
                name_rva = struct.unpack_from("<I", data, off + 20)[0]
                name_len = struct.unpack_from("<I", data, name_rva)[0]
                name = data[name_rva + 4 : name_rva + 4 + name_len].decode(
                    "utf-16-le", errors="replace"
                )
                if module_name in name:
                    return size_of_image
        return None

    def get_expected_size(self, target, module):
        """Compute expected SizeOfImage the same way getModuleFileSize does:
        start with the first segment's byte size, then walk contiguous leaf
        (deepest child) sections."""
        first = module.GetSectionAtIndex(0)
        first_addr = first.GetLoadAddress(target)
        total = first.GetByteSize()
        next_addr = first_addr + total

        # Build a map of leaf-section-address → size
        leaves = {}
        for i in range(module.GetNumSections()):
            sec = module.GetSectionAtIndex(i)
            if sec.GetNumSubSections() == 0:
                a = sec.GetLoadAddress(target)
                if a != lldb.LLDB_INVALID_ADDRESS:
                    leaves[a] = sec.GetByteSize()
            else:
                for j in range(sec.GetNumSubSections()):
                    c = sec.GetSubSectionAtIndex(j)
                    a = c.GetLoadAddress(target)
                    if a != lldb.LLDB_INVALID_ADDRESS:
                        leaves[a] = c.GetByteSize()

        while next_addr in leaves:
            total += leaves[next_addr]
            next_addr += leaves[next_addr]

        return total

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_size_of_image_with_contiguous_segments(self):
        """Test SizeOfImage is correct for a module with contiguous segments."""
        self.build()
        testlib_path = self.build_shared_lib()
        exe = self.getBuildArtifact("a.out")
        core_path = self.getBuildArtifact("core.dmp")

        target = self.dbg.CreateTarget(exe)
        process = target.LaunchSimple(
            [testlib_path], None, self.get_process_working_directory()
        )
        self.assertState(process.GetState(), lldb.eStateStopped)

        testlib = target.FindModule(lldb.SBFileSpec("libtestlib.so"))
        self.assertTrue(testlib.IsValid(), "libtestlib.so not loaded")

        expected = self.get_expected_size(target, testlib)

        options = lldb.SBSaveCoreOptions()
        options.SetOutputFile(lldb.SBFileSpec(core_path))
        options.SetPluginName("minidump")
        options.SetStyle(lldb.eSaveCoreStackOnly)
        error = process.SaveCore(options)
        self.assertTrue(error.Success(), error.GetCString())

        actual = self.get_minidump_size_of_image(core_path, "libtestlib")
        self.assertIsNotNone(actual, "libtestlib not found in minidump")
        self.assertEqual(
            actual, expected, f"SizeOfImage: got {actual:#x}, want {expected:#x}"
        )

        self.assertSuccess(process.Kill())
        self.dbg.DeleteTarget(target)
