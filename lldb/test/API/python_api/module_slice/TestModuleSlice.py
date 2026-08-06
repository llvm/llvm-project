"""
Test adding a module that is a slice of a containing file.

A HIP binary carries its device code inside a section of the host executable,
reported to the debugger as a file path plus a byte offset and size. A
ModuleSpec with those bounds has to read the embedded object, not the file.
"""

import binascii

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

# A host executable carrying a device code object in a .hip_fatbin section.
FAT_BINARY_YAML = """\
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x1000
    Content:         "c3"
  - Name:            .hip_fatbin
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC ]
    AddressAlign:    0x1000
    Content:         "%s"
"""


class ModuleSliceTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def make_gpu_code_object(self):
        path = self.getBuildArtifact("gpu_code_object.so")
        self.yaml2obj("gpu_code_object.yaml", path)
        return path

    def make_fat_binary(self, gpu_path):
        """Embed the device code object the way a HIP binary carries it."""
        with open(gpu_path, "rb") as f:
            content = binascii.hexlify(f.read()).decode()
        yaml_path = self.getBuildArtifact("fat_binary.yaml")
        with open(yaml_path, "w") as f:
            f.write(FAT_BINARY_YAML % content)
        container = self.getBuildArtifact("fat_binary")
        self.yaml2obj(yaml_path, container)
        return container

    def fatbin_section_range(self, container):
        """Locate the embedded code object by the bounds of its section."""
        target = self.dbg.CreateTarget("")
        spec = lldb.SBModuleSpec()
        spec.SetFileSpec(lldb.SBFileSpec(container))
        module = target.AddModule(spec)
        self.assertTrue(module.IsValid(), "the host executable should load")

        section = module.FindSection(".hip_fatbin")
        self.assertTrue(section.IsValid(), "host binary should have .hip_fatbin")
        offset, size = section.GetFileOffset(), section.GetFileByteSize()
        self.dbg.DeleteTarget(target)
        return offset, size

    def text_file_address(self, module):
        section = module.FindSection(".text")
        self.assertTrue(section.IsValid(), "module should have a .text section")
        return section.GetFileAddress()

    def test_code_object_in_fat_binary(self):
        """The code object inside a .hip_fatbin section is read, not the host."""
        gpu = self.make_gpu_code_object()
        container = self.make_fat_binary(gpu)
        offset, size = self.fatbin_section_range(container)

        target = self.dbg.CreateTarget("")
        spec = lldb.SBModuleSpec()
        spec.SetFileSpec(lldb.SBFileSpec(container))
        spec.SetObjectOffset(offset)
        spec.SetObjectSize(size)

        module = target.AddModule(spec)
        self.assertTrue(module.IsValid(), "the code object should load")
        # The device code object has .text at 0x3000, the host at 0x1000.
        self.assertEqual(self.text_file_address(module), 0x3000)

    def test_whole_file(self):
        """Without an object offset the containing file itself is read."""
        gpu = self.make_gpu_code_object()
        container = self.make_fat_binary(gpu)

        target = self.dbg.CreateTarget("")
        spec = lldb.SBModuleSpec()
        spec.SetFileSpec(lldb.SBFileSpec(container))

        module = target.AddModule(spec)
        self.assertTrue(module.IsValid(), "the host executable should load")
        self.assertEqual(self.text_file_address(module), 0x1000)
