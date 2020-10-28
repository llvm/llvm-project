//===-- ModuleSpecTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Utility/DataBuffer.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

extern const char *TestMainArgv0;

// This test file intentionally doesn't initialize the FileSystem.
// Everything in this file should be able to run without requiring
// any interaction with the FileSystem class; by keeping it
// uninitialized, it will assert if anything tries to interact with
// it.

TEST(ModuleSpecTest, InvalidInMemoryBuffer) {
  uint8_t Invalid[] = "This is not a binary file.";
  DataBufferSP InvalidBufferSP =
      std::make_shared<DataBufferUnowned>(Invalid, sizeof(Invalid));
  ModuleSpec Spec(FileSpec(), UUID(), InvalidBufferSP);

  auto InvalidModuleSP = std::make_shared<Module>(Spec);
  ASSERT_EQ(InvalidModuleSP->GetObjectFile(), nullptr);
}

TEST(ModuleSpecTest, InvalidInMemoryBufferValidFile) {
  uint8_t Invalid[] = "This is not a binary file.";
  DataBufferSP InvalidBufferSP =
      std::make_shared<DataBufferUnowned>(Invalid, sizeof(Invalid));
  ModuleSpec Spec(FileSpec(TestMainArgv0), UUID(), InvalidBufferSP);

  auto InvalidModuleSP = std::make_shared<Module>(Spec);
  ASSERT_EQ(InvalidModuleSP->GetObjectFile(), nullptr);
}

TEST(ModuleSpecTest, TestELFFile) {
  SubsystemRAII<ObjectFileELF> subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_REL
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto M = std::make_shared<Module>(ExpectedFile->moduleSpec());
  ObjectFile *OF = M->GetObjectFile();

  ASSERT_EQ(llvm::isa<ObjectFileELF>(OF), true);
}

TEST(ModuleSpecTest, TestCOFFFile) {
  SubsystemRAII<ObjectFilePECOFF> subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !COFF
OptionalHeader:
  AddressOfEntryPoint: 0
  ImageBase:       16777216
  SectionAlignment: 4096
  FileAlignment:   512
  MajorOperatingSystemVersion: 6
  MinorOperatingSystemVersion: 0
  MajorImageVersion: 0
  MinorImageVersion: 0
  MajorSubsystemVersion: 6
  MinorSubsystemVersion: 0
  Subsystem:       IMAGE_SUBSYSTEM_WINDOWS_CUI
  DLLCharacteristics: [ IMAGE_DLL_CHARACTERISTICS_HIGH_ENTROPY_VA, IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE, IMAGE_DLL_CHARACTERISTICS_NX_COMPAT ]
  SizeOfStackReserve: 1048576
  SizeOfStackCommit: 4096
  SizeOfHeapReserve: 1048576
  SizeOfHeapCommit: 4096
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [ IMAGE_FILE_EXECUTABLE_IMAGE, IMAGE_FILE_LARGE_ADDRESS_AWARE ]
sections:
  - Name:            .text
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    VirtualAddress:  4096
    VirtualSize:     4096
symbols:         []
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto M = std::make_shared<Module>(ExpectedFile->moduleSpec());
  ObjectFile *OF = M->GetObjectFile();

  ASSERT_EQ(llvm::isa<ObjectFilePECOFF>(OF), true);
}

TEST(ModuleSpecTest, TestMachOFile) {
  SubsystemRAII<ObjectFileMachO> subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x0100000C
  cpusubtype:      0x00000000
  filetype:        0x00000001
  ncmds:           1
  sizeofcmds:      232
  flags:           0x00002000
  reserved:        0x00000000
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         232
    segname:         ''
    vmaddr:          0
    vmsize:          56
    fileoff:         392
    filesize:        56
    maxprot:         7
    initprot:        7
    nsects:          1
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0000000000000000
        size:            24
        offset:          0x00000188
        align:           2
        reloff:          0x00000000
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x00000000
        reserved2:       0x00000000
        reserved3:       0x00000000
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto M = std::make_shared<Module>(ExpectedFile->moduleSpec());
  ObjectFile *OF = M->GetObjectFile();

  ASSERT_EQ(llvm::isa<ObjectFileMachO>(OF), true);
}
