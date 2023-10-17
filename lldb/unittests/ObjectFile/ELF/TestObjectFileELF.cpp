//===-- TestObjectFileELF.cpp ---------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

class ObjectFileELFTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab>
      subsystems;
};

TEST_F(ObjectFileELFTest, SectionsResolveConsistently) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
  Entry:           0x0000000000400180
Sections:
  - Name:            .note.gnu.build-id
    Type:            SHT_NOTE
    Flags:           [ SHF_ALLOC ]
    Address:         0x0000000000400158
    AddressAlign:    0x0000000000000004
    Content:         040000001400000003000000474E55003F3EC29E3FD83E49D18C4D49CD8A730CC13117B6
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000400180
    AddressAlign:    0x0000000000000010
    Content:         554889E58B042500106000890425041060005DC3
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000601000
    AddressAlign:    0x0000000000000004
    Content:         2F000000
  - Name:            .bss
    Type:            SHT_NOBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000601004
    AddressAlign:    0x0000000000000004
    Size:            0x0000000000000004
Symbols:
  - Name:            Y
    Type:            STT_OBJECT
    Section:         .data
    Value:           0x0000000000601000
    Size:            0x0000000000000004
    Binding:         STB_GLOBAL
  - Name:            _start
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000400180
    Size:            0x0000000000000014
    Binding:         STB_GLOBAL
  - Name:            X
    Type:            STT_OBJECT
    Section:         .bss
    Value:           0x0000000000601004
    Size:            0x0000000000000004
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto bss_sp = list->FindSectionByName(ConstString(".bss"));
  ASSERT_NE(nullptr, bss_sp);
  auto data_sp = list->FindSectionByName(ConstString(".data"));
  ASSERT_NE(nullptr, data_sp);
  auto text_sp = list->FindSectionByName(ConstString(".text"));
  ASSERT_NE(nullptr, text_sp);

  const Symbol *X = module_sp->FindFirstSymbolWithNameAndType(ConstString("X"),
                                                              eSymbolTypeAny);
  ASSERT_NE(nullptr, X);
  EXPECT_EQ(bss_sp, X->GetAddress().GetSection());

  const Symbol *Y = module_sp->FindFirstSymbolWithNameAndType(ConstString("Y"),
                                                              eSymbolTypeAny);
  ASSERT_NE(nullptr, Y);
  EXPECT_EQ(data_sp, Y->GetAddress().GetSection());

  const Symbol *start = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("_start"), eSymbolTypeAny);
  ASSERT_NE(nullptr, start);
  EXPECT_EQ(text_sp, start->GetAddress().GetSection());
}

// Test that GetModuleSpecifications works on an "atypical" object file which
// has section headers right after the ELF header (instead of the more common
// layout where the section headers are at the very end of the object file).
//
// Test file generated with yaml2obj (@svn rev 324254) from the following input:
/*
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
  Entry:           0x00000000004003D0
Sections:
  - Name:            .note.gnu.build-id
    Type:            SHT_NOTE
    Flags:           [ SHF_ALLOC ]
    Address:         0x0000000000400274
    AddressAlign:    0x0000000000000004
    Content:         040000001400000003000000474E55001B8A73AC238390E32A7FF4AC8EBE4D6A41ECF5C9
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x00000000004003D0
    AddressAlign:    0x0000000000000010
    Content:         DEADBEEFBAADF00D
...
*/
TEST_F(ObjectFileELFTest, GetModuleSpecifications_EarlySectionHeaders) {
  std::string SO = GetInputFilePath("early-section-headers.so");
  ModuleSpecList Specs;
  ASSERT_EQ(1u, ObjectFile::GetModuleSpecifications(FileSpec(SO), 0, 0, Specs));
  ModuleSpec Spec;
  ASSERT_TRUE(Specs.GetModuleSpecAtIndex(0, Spec)) ;
  UUID Uuid;
  Uuid.SetFromStringRef("1b8a73ac238390e32a7ff4ac8ebe4d6a41ecf5c9");
  EXPECT_EQ(Spec.GetUUID(), Uuid);
}

TEST_F(ObjectFileELFTest, GetModuleSpecifications_OffsetSizeWithNormalFile) {
  std::string SO = GetInputFilePath("liboffset-test.so");
  ModuleSpecList Specs;
  ASSERT_EQ(1u, ObjectFile::GetModuleSpecifications(FileSpec(SO), 0, 0, Specs));
  ModuleSpec Spec;
  ASSERT_TRUE(Specs.GetModuleSpecAtIndex(0, Spec)) ;
  UUID Uuid;
  Uuid.SetFromStringRef("7D6E4738");
  EXPECT_EQ(Spec.GetUUID(), Uuid);
  EXPECT_EQ(Spec.GetObjectOffset(), 0UL);
  EXPECT_EQ(Spec.GetObjectSize(), 3600UL);
  EXPECT_EQ(FileSystem::Instance().GetByteSize(FileSpec(SO)), 3600UL);
}

TEST_F(ObjectFileELFTest, GetModuleSpecifications_OffsetSizeWithOffsetFile) {
  // The contents of offset-test.bin are
  // -    0-1023: \0
  // - 1024-4623: liboffset-test.so (offset: 1024, size: 3600, CRC32: 7D6E4738)
  // - 4624-4639: \0
  std::string SO = GetInputFilePath("offset-test.bin");
  ModuleSpecList Specs;
  ASSERT_EQ(
      1u, ObjectFile::GetModuleSpecifications(FileSpec(SO), 1024, 3600, Specs));
  ModuleSpec Spec;
  ASSERT_TRUE(Specs.GetModuleSpecAtIndex(0, Spec)) ;
  UUID Uuid;
  Uuid.SetFromStringRef("7D6E4738");
  EXPECT_EQ(Spec.GetUUID(), Uuid);
  EXPECT_EQ(Spec.GetObjectOffset(), 1024UL);
  EXPECT_EQ(Spec.GetObjectSize(), 3600UL);
  EXPECT_EQ(FileSystem::Instance().GetByteSize(FileSpec(SO)), 4640UL);
}

TEST_F(ObjectFileELFTest, GetSymtab_NoSymEntryPointArmThumbAddressClass) {
  /*
  // nosym-entrypoint-arm-thumb.s
  .thumb_func
  _start:
      mov r0, #42
      mov r7, #1
      svc #0
  // arm-linux-androideabi-as nosym-entrypoint-arm-thumb.s
  //   -o nosym-entrypoint-arm-thumb.o
  // arm-linux-androideabi-ld nosym-entrypoint-arm-thumb.o
  //   -o nosym-entrypoint-arm-thumb -e 0x8075 -s
  */
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS32
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_ARM
  Flags:           [ EF_ARM_SOFT_FLOAT, EF_ARM_EABI_VER5 ]
  Entry:           0x0000000000008075
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000008074
    AddressAlign:    0x0000000000000002
    Content:         2A20012700DF
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000009000
    AddressAlign:    0x0000000000000001
    Content:         ''
  - Name:            .bss
    Type:            SHT_NOBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000009000
    AddressAlign:    0x0000000000000001
  - Name:            .note.gnu.gold-version
    Type:            SHT_NOTE
    AddressAlign:    0x0000000000000004
    Content:         040000000900000004000000474E5500676F6C6420312E3131000000
  - Name:            .ARM.attributes
    Type:            SHT_ARM_ATTRIBUTES
    AddressAlign:    0x0000000000000001
    Content:         '4113000000616561626900010900000006020901'
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  auto entry_point_addr = module_sp->GetObjectFile()->GetEntryPointAddress();
  ASSERT_TRUE(entry_point_addr.GetOffset() & 1);
  // Decrease the offsite by 1 to make it into a breakable address since this
  // is Thumb.
  entry_point_addr.SetOffset(entry_point_addr.GetOffset() - 1);
  ASSERT_EQ(entry_point_addr.GetAddressClass(),
            AddressClass::eCodeAlternateISA);
}

TEST_F(ObjectFileELFTest, GetSymtab_NoSymEntryPointArmAddressClass) {
  /*
  // nosym-entrypoint-arm.s
  _start:
      movs r0, #42
      movs r7, #1
      svc #0
  // arm-linux-androideabi-as nosym-entrypoint-arm.s
  //   -o nosym-entrypoint-arm.o
  // arm-linux-androideabi-ld nosym-entrypoint-arm.o
  //   -o nosym-entrypoint-arm -e 0x8074 -s
  */
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS32
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_ARM
  Flags:           [ EF_ARM_SOFT_FLOAT, EF_ARM_EABI_VER5 ]
  Entry:           0x0000000000008074
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000008074
    AddressAlign:    0x0000000000000004
    Content:         2A00A0E30170A0E3000000EF
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000009000
    AddressAlign:    0x0000000000000001
    Content:         ''
  - Name:            .bss
    Type:            SHT_NOBITS
    Flags:           [ SHF_WRITE, SHF_ALLOC ]
    Address:         0x0000000000009000
    AddressAlign:    0x0000000000000001
  - Name:            .note.gnu.gold-version
    Type:            SHT_NOTE
    AddressAlign:    0x0000000000000004
    Content:         040000000900000004000000474E5500676F6C6420312E3131000000
  - Name:            .ARM.attributes
    Type:            SHT_ARM_ATTRIBUTES
    AddressAlign:    0x0000000000000001
    Content:         '4113000000616561626900010900000006010801'
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());

  auto entry_point_addr = module_sp->GetObjectFile()->GetEntryPointAddress();
  ASSERT_EQ(entry_point_addr.GetAddressClass(), AddressClass::eCode);
}
