//===-- TestDWARFCallFrameInfo.cpp ----------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Process/Utility/RegisterContext_x86.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/Testing/Support/Error.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb_private;
using namespace lldb;

class DWARFCallFrameInfoTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab>
      subsystems;

protected:
  void TestBasic(DWARFCallFrameInfo::Type type, llvm::StringRef symbol);
  void TestValOffset(DWARFCallFrameInfo::Type type, llvm::StringRef symbol);
};

namespace lldb_private {
static std::ostream &operator<<(std::ostream &OS, const UnwindPlan::Row &row) {
  StreamString SS;
  row.Dump(SS, nullptr, nullptr, 0);
  return OS << SS.GetData();
}
} // namespace lldb_private

static UnwindPlan::Row GetExpectedRow0() {
  UnwindPlan::Row row;
  row.SetOffset(0);
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_rsp_x86_64, 8);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rip_x86_64, -8, false);
  return row;
}

static UnwindPlan::Row GetExpectedRow1() {
  UnwindPlan::Row row;
  row.SetOffset(1);
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_rsp_x86_64, 16);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rip_x86_64, -8, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rbp_x86_64, -16, false);
  return row;
}

static UnwindPlan::Row GetExpectedRow2() {
  UnwindPlan::Row row;
  row.SetOffset(4);
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_rbp_x86_64, 16);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rip_x86_64, -8, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rbp_x86_64, -16, false);
  return row;
}

void DWARFCallFrameInfoTest::TestBasic(DWARFCallFrameInfo::Type type,
                                       llvm::StringRef symbol) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
  Entry:           0x0000000000000260
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000000260
    AddressAlign:    0x0000000000000010
    Content:         554889E5897DFC8B45FC5DC30F1F4000554889E5897DFC8B45FC5DC30F1F4000554889E5897DFC8B45FC5DC3
#0000000000000260 <eh_frame>:
# 260:	55                   	push   %rbp
# 261:	48 89 e5             	mov    %rsp,%rbp
# 264:	89 7d fc             	mov    %edi,-0x4(%rbp)
# 267:	8b 45 fc             	mov    -0x4(%rbp),%eax
# 26a:	5d                   	pop    %rbp
# 26b:	c3                   	retq
# 26c:	0f 1f 40 00          	nopl   0x0(%rax)
#
#0000000000000270 <debug_frame3>:
# 270:	55                   	push   %rbp
# 271:	48 89 e5             	mov    %rsp,%rbp
# 274:	89 7d fc             	mov    %edi,-0x4(%rbp)
# 277:	8b 45 fc             	mov    -0x4(%rbp),%eax
# 27a:	5d                   	pop    %rbp
# 27b:	c3                   	retq
# 27c:	0f 1f 40 00          	nopl   0x0(%rax)
#
#0000000000000280 <debug_frame4>:
# 280:	55                   	push   %rbp
# 281:	48 89 e5             	mov    %rsp,%rbp
# 284:	89 7d fc             	mov    %edi,-0x4(%rbp)
# 287:	8b 45 fc             	mov    -0x4(%rbp),%eax
# 28a:	5d                   	pop    %rbp
# 28b:	c3                   	retq
  - Name:            .eh_frame
    Type:            SHT_X86_64_UNWIND
    Flags:           [ SHF_ALLOC ]
    Address:         0x0000000000000290
    AddressAlign:    0x0000000000000008
    Content:         1400000000000000017A5200017810011B0C0708900100001C0000001C000000B0FFFFFF0C00000000410E108602430D0600000000000000
#00000000 0000000000000014 00000000 CIE
#  Version:               1
#  Augmentation:          "zR"
#  Code alignment factor: 1
#  Data alignment factor: -8
#  Return address column: 16
#  Augmentation data:     1b
#
#  DW_CFA_def_cfa: r7 (rsp) ofs 8
#  DW_CFA_offset: r16 (rip) at cfa-8
#  DW_CFA_nop
#  DW_CFA_nop
#
#00000018 000000000000001c 0000001c FDE cie=00000000 pc=ffffffffffffffd0..ffffffffffffffdc
#  DW_CFA_advance_loc: 1 to ffffffffffffffd1
#  DW_CFA_def_cfa_offset: 16
#  DW_CFA_offset: r6 (rbp) at cfa-16
#  DW_CFA_advance_loc: 3 to ffffffffffffffd4
#  DW_CFA_def_cfa_register: r6 (rbp)
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
  - Name:            .debug_frame
    Type:            SHT_PROGBITS
    AddressAlign:    0x0000000000000008
    Content:         14000000FFFFFFFF03000178100C070890010000000000001C0000000000000070020000000000000C00000000000000410E108602430D0614000000FFFFFFFF040008000178100C07089001000000001C0000003800000080020000000000000C00000000000000410E108602430D06
#00000000 0000000000000014 ffffffff CIE
#  Version:               3
#  Augmentation:          ""
#  Code alignment factor: 1
#  Data alignment factor: -8
#  Return address column: 16
#
#  DW_CFA_def_cfa: r7 (rsp) ofs 8
#  DW_CFA_offset: r16 (rip) at cfa-8
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#
#00000018 000000000000001c 00000000 FDE cie=00000000 pc=0000000000000270..000000000000027c
#  DW_CFA_advance_loc: 1 to 0000000000000271
#  DW_CFA_def_cfa_offset: 16
#  DW_CFA_offset: r6 (rbp) at cfa-16
#  DW_CFA_advance_loc: 3 to 0000000000000274
#  DW_CFA_def_cfa_register: r6 (rbp)
#
#00000038 0000000000000014 ffffffff CIE
#  Version:               4
#  Augmentation:          ""
#  Pointer Size:          8
#  Segment Size:          0
#  Code alignment factor: 1
#  Data alignment factor: -8
#  Return address column: 16
#
#  DW_CFA_def_cfa: r7 (rsp) ofs 8
#  DW_CFA_offset: r16 (rip) at cfa-8
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#  DW_CFA_nop
#
#00000050 000000000000001c 00000038 FDE cie=00000038 pc=0000000000000280..000000000000028c
#  DW_CFA_advance_loc: 1 to 0000000000000281
#  DW_CFA_def_cfa_offset: 16
#  DW_CFA_offset: r6 (rbp) at cfa-16
#  DW_CFA_advance_loc: 3 to 0000000000000284
#  DW_CFA_def_cfa_register: r6 (rbp)
Symbols:
  - Name:            eh_frame
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000000260
    Size:            0x000000000000000C
    Binding:         STB_GLOBAL
  - Name:            debug_frame3
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000000270
    Size:            0x000000000000000C
    Binding:         STB_GLOBAL
  - Name:            debug_frame4
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000000280
    Size:            0x000000000000000C
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto section_sp = list->FindSectionByType(type == DWARFCallFrameInfo::EH
                                                ? eSectionTypeEHFrame
                                                : eSectionTypeDWARFDebugFrame,
                                            false);
  ASSERT_NE(nullptr, section_sp);

  DWARFCallFrameInfo cfi(*module_sp->GetObjectFile(), section_sp, type);

  const Symbol *sym = module_sp->FindFirstSymbolWithNameAndType(
      ConstString(symbol), eSymbolTypeAny);
  ASSERT_NE(nullptr, sym);

  std::unique_ptr<UnwindPlan> plan_up = cfi.GetUnwindPlan(sym->GetAddress());
  ASSERT_TRUE(plan_up);
  ASSERT_EQ(3, plan_up->GetRowCount());
  EXPECT_THAT(plan_up->GetRowAtIndex(0), testing::Pointee(GetExpectedRow0()));
  EXPECT_THAT(plan_up->GetRowAtIndex(1), testing::Pointee(GetExpectedRow1()));
  EXPECT_THAT(plan_up->GetRowAtIndex(2), testing::Pointee(GetExpectedRow2()));
}

TEST_F(DWARFCallFrameInfoTest, Basic_dwarf3) {
  TestBasic(DWARFCallFrameInfo::DWARF, "debug_frame3");
}

TEST_F(DWARFCallFrameInfoTest, Basic_dwarf4) {
  TestBasic(DWARFCallFrameInfo::DWARF, "debug_frame4");
}

TEST_F(DWARFCallFrameInfoTest, Basic_eh) {
  TestBasic(DWARFCallFrameInfo::EH, "eh_frame");
}

static UnwindPlan::Row GetValOffsetExpectedRow0() {
  UnwindPlan::Row row;
  row.SetOffset(0);
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_rsp_x86_64, 16);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rip_x86_64, -8, false);
  row.SetRegisterLocationToIsCFAPlusOffset(dwarf_rbp_x86_64, -16, false);
  return row;
}

void DWARFCallFrameInfoTest::TestValOffset(DWARFCallFrameInfo::Type type,
                                           llvm::StringRef symbol) {
  // This test is artificial as X86 does not use DW_CFA_val_offset but this
  // test verifies that we can successfully interpret them if they do occur.
  // Note the distinction between RBP and RIP in this part of the DWARF dump:
  // 0x0: CFA=RSP+16: RBP=CFA-16, RIP=[CFA-8]
  // Whereas RIP is stored in the memory CFA-8 points at, RBP is reconstructed
  // from the CFA without any memory access.
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_REL
  Machine:         EM_X86_64
  SectionHeaderStringTable: .strtab
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x4
    Content:         0F1F00
  - Name:            .debug_frame
    Type:            SHT_PROGBITS
    AddressAlign:    0x8
#00000000 00000014 ffffffff CIE
#  Format:                DWARF32
#  Version:               4
#  Augmentation:          ""
#  Address size:          8
#  Segment desc size:     0
#  Code alignment factor: 1
#  Data alignment factor: -8
#  Return address column: 16
#
#  DW_CFA_def_cfa: RSP +8
#  DW_CFA_offset: RIP -8
#  DW_CFA_nop:
#  DW_CFA_nop:
#  DW_CFA_nop:
#  DW_CFA_nop:
#
#  CFA=RSP+8: RIP=[CFA-8]
#
#00000018 0000001c 00000000 FDE cie=00000000 pc=00000000...00000003
#  Format:       DWARF32
#  DW_CFA_def_cfa_offset: +16
#  DW_CFA_val_offset: RBP -16
#  DW_CFA_nop:
#  DW_CFA_nop:
#  DW_CFA_nop:
#
#  0x0: CFA=RSP+16: RBP=CFA-16, RIP=[CFA-8]
    Content:         14000000FFFFFFFF040008000178100C07089001000000001C00000000000000000000000000000003000000000000000E10140602000000
  - Name:            .rela.debug_frame
    Type:            SHT_RELA
    Flags:           [ SHF_INFO_LINK ]
    Link:            .symtab
    AddressAlign:    0x8
    Info:            .debug_frame
    Relocations:
      - Offset:          0x1C
        Symbol:          .debug_frame
        Type:            R_X86_64_32
      - Offset:          0x20
        Symbol:          .text
        Type:            R_X86_64_64
  - Type:            SectionHeaderTable
    Sections:
      - Name:            .strtab
      - Name:            .text
      - Name:            .debug_frame
      - Name:            .rela.debug_frame
      - Name:            .symtab
Symbols:
  - Name:            .text
    Type:            STT_SECTION
    Section:         .text
  - Name:            debug_frame3
    Section:         .text
  - Name:            .debug_frame
    Type:            STT_SECTION
    Section:         .debug_frame
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto section_sp = list->FindSectionByType(type == DWARFCallFrameInfo::EH
                                                ? eSectionTypeEHFrame
                                                : eSectionTypeDWARFDebugFrame,
                                            false);
  ASSERT_NE(nullptr, section_sp);

  DWARFCallFrameInfo cfi(*module_sp->GetObjectFile(), section_sp, type);

  const Symbol *sym = module_sp->FindFirstSymbolWithNameAndType(
      ConstString(symbol), eSymbolTypeAny);
  ASSERT_NE(nullptr, sym);

  std::unique_ptr<UnwindPlan> plan_up = cfi.GetUnwindPlan(sym->GetAddress());
  ASSERT_TRUE(plan_up);
  ASSERT_EQ(1, plan_up->GetRowCount());
  EXPECT_THAT(plan_up->GetRowAtIndex(0),
              testing::Pointee(GetValOffsetExpectedRow0()));
}

TEST_F(DWARFCallFrameInfoTest, ValOffset_dwarf3) {
  TestValOffset(DWARFCallFrameInfo::DWARF, "debug_frame3");
}

// Test that we correctly handle invalid FDE entries that have CIE ID values
TEST_F(DWARFCallFrameInfoTest, InvalidFDEWithCIEID_dwarf32) {
  // Create an FDE with cie_offset of 0xFFFFFFFF (DW_CIE_ID) which is invalid
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
    Address:         0x0000000000000260
    AddressAlign:    0x0000000000000010
    Content:         554889E5897DFC8B45FC5DC3
  - Name:            .debug_frame
    Type:            SHT_PROGBITS
    AddressAlign:    0x0000000000000008
    # First, a valid CIE
    # 00000000 0000000000000014 ffffffff CIE
    #   Version:               3
    #   Augmentation:          ""
    #   Code alignment factor: 1
    #   Data alignment factor: -8
    #   Return address column: 16
    Content:         14000000FFFFFFFF03000178100C0708900100000000000018000000FFFFFFFF60020000000000000C00000000000000
    # Then an invalid FDE with CIE pointer = 0xFFFFFFFF (which would make it look like a CIE)
    # 00000018 0000000000000018 ffffffff FDE cie=ffffffff pc=0000000000000260..000000000000026c
    # The cie offset of 0xFFFFFFFF is invalid for an FDE in debug_frame
Symbols:
  - Name:            test_invalid
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000000260
    Size:            0x000000000000000C
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto section_sp = list->FindSectionByType(eSectionTypeDWARFDebugFrame, false);
  ASSERT_NE(nullptr, section_sp);

  DWARFCallFrameInfo cfi(*module_sp->GetObjectFile(), section_sp,
                         DWARFCallFrameInfo::DWARF);

  // This should trigger our assertion or return nullptr because the FDE is
  // invalid
  const Symbol *sym = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("test_invalid"), eSymbolTypeAny);
  ASSERT_NE(nullptr, sym);

  std::unique_ptr<UnwindPlan> plan_up = cfi.GetUnwindPlan(sym->GetAddress());
  // The plan should be null because we have an invalid FDE
  EXPECT_EQ(nullptr, plan_up);
}

// Test that we correctly handle invalid FDE entries that have CIE ID values
TEST_F(DWARFCallFrameInfoTest, InvalidFDEWithCIEID_dwarf64) {
  // Create an FDE with cie_offset of 0xFFFFFFFFFFFFFFFF (DW64_CIE_ID) which is
  // invalid
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
    Address:         0x0000000000000260
    AddressAlign:    0x0000000000000010
    Content:         554889E5897DFC8B45FC5DC3
  - Name:            .debug_frame
    Type:            SHT_PROGBITS
    AddressAlign:    0x0000000000000008
    # DWARF64 format CIE
    # Initial length: 0xFFFFFFFF followed by 64-bit length
    # 00000000 ffffffff 0000000000000014 ffffffffffffffff CIE
    Content:         FFFFFFFF1400000000000000FFFFFFFFFFFFFFFF03000178100C0708900100000000FFFFFFFF1800000000000000FFFFFFFFFFFFFFFF60020000000000000C00000000000000
    # DWARF64 FDE with invalid CIE pointer = 0xFFFFFFFFFFFFFFFF
    # Initial length: 0xFFFFFFFF, followed by 64-bit length (0x18)
    # Then 64-bit CIE pointer: 0xFFFFFFFFFFFFFFFF (which is DW64_CIE_ID, invalid for FDE)
Symbols:
  - Name:            test_invalid64
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000000260
    Size:            0x000000000000000C
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto section_sp = list->FindSectionByType(eSectionTypeDWARFDebugFrame, false);
  ASSERT_NE(nullptr, section_sp);

  DWARFCallFrameInfo cfi(*module_sp->GetObjectFile(), section_sp,
                         DWARFCallFrameInfo::DWARF);

  const Symbol *sym = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("test_invalid64"), eSymbolTypeAny);
  ASSERT_NE(nullptr, sym);

  std::unique_ptr<UnwindPlan> plan_up = cfi.GetUnwindPlan(sym->GetAddress());
  // The plan should be null because we have an invalid FDE
  EXPECT_EQ(nullptr, plan_up);
}

// Test valid CIE markers in eh_frame format
TEST_F(DWARFCallFrameInfoTest, ValidCIEMarkers_eh_frame) {
  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
  Entry:           0x0000000000000260
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0000000000000260
    AddressAlign:    0x0000000000000010
    Content:         554889E5897DFC8B45FC5DC3
  - Name:            .eh_frame
    Type:            SHT_X86_64_UNWIND
    Flags:           [ SHF_ALLOC ]
    Address:         0x0000000000000290
    AddressAlign:    0x0000000000000008
    # eh_frame content
    # CIE + FDE that works with address 0x260
    Content:         1400000000000000017A5200017810011B0C0708900100001C0000001C000000B0FFFFFF0C00000000410E108602430D0600000000000000
Symbols:
  - Name:            simple_function
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000000260
    Size:            0x000000000000000F
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto section_sp = list->FindSectionByType(eSectionTypeEHFrame, false);
  ASSERT_NE(nullptr, section_sp);

  DWARFCallFrameInfo cfi(*module_sp->GetObjectFile(), section_sp,
                         DWARFCallFrameInfo::EH);

  const Symbol *sym = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("simple_function"), eSymbolTypeAny);
  ASSERT_NE(nullptr, sym);

  std::unique_ptr<UnwindPlan> plan_up = cfi.GetUnwindPlan(sym->GetAddress());
  // Should succeed with valid CIE and FDE
  ASSERT_NE(nullptr, plan_up);
  EXPECT_GE(plan_up->GetRowCount(), 1);
}

// Test valid CIE markers in debug_frame DWARF32 format
TEST_F(DWARFCallFrameInfoTest, ValidCIEMarkers_dwarf32) {
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
    Address:         0x0000000000001130
    AddressAlign:    0x0000000000000010
    Content:         554889E5897DFC8B45FC83C0015DC3
  - Name:            .debug_frame
    Type:            SHT_PROGBITS
    AddressAlign:    0x0000000000000008
    # debug_frame content in DWARF32 format
    # CIE (length=0x14, CIE_id=0xFFFFFFFF, version=4)
    # FDE (length=0x24, CIE_offset=0)
    Content:         14000000FFFFFFFF040008000178100C0708900100000000240000000000000030110000000000000F00000000000000410E108602430D064A0C070800000000
Symbols:
  - Name:            simple_function
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000001130
    Size:            0x000000000000000F
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto section_sp = list->FindSectionByType(eSectionTypeDWARFDebugFrame, false);
  ASSERT_NE(nullptr, section_sp);

  DWARFCallFrameInfo cfi(*module_sp->GetObjectFile(), section_sp,
                         DWARFCallFrameInfo::DWARF);

  const Symbol *sym = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("simple_function"), eSymbolTypeAny);
  ASSERT_NE(nullptr, sym);

  std::unique_ptr<UnwindPlan> plan_up = cfi.GetUnwindPlan(sym->GetAddress());
  // Should succeed with valid CIE and FDE
  ASSERT_NE(nullptr, plan_up);
  EXPECT_GE(plan_up->GetRowCount(), 1);
}

// Test valid CIE markers in debug_frame DWARF64 format
TEST_F(DWARFCallFrameInfoTest, ValidCIEMarkers_dwarf64) {
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
    Address:         0x0000000000001130
    AddressAlign:    0x0000000000000010
    Content:         554889E5897DFC8B45FC83C0015DC3
  - Name:            .debug_frame
    Type:            SHT_PROGBITS
    AddressAlign:    0x0000000000000008
    # debug_frame content in DWARF64 format
    # CIE: length_marker=0xFFFFFFFF, length=0x14, CIE_id=0xFFFFFFFFFFFFFFFF, version=4
    # FDE: length_marker=0xFFFFFFFF, length=0x24, CIE_offset=0x0 (points to CIE)
    Content:         FFFFFFFF1400000000000000FFFFFFFFFFFFFFFF040008000178100C07089001FFFFFFFF2400000000000000000000000000000030110000000000000F00000000000000410E108602430D064A0C0708
Symbols:
  - Name:            simple_function
    Type:            STT_FUNC
    Section:         .text
    Value:           0x0000000000001130
    Size:            0x000000000000000F
    Binding:         STB_GLOBAL
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(ExpectedFile->moduleSpec());
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto section_sp = list->FindSectionByType(eSectionTypeDWARFDebugFrame, false);
  ASSERT_NE(nullptr, section_sp);

  DWARFCallFrameInfo cfi(*module_sp->GetObjectFile(), section_sp,
                         DWARFCallFrameInfo::DWARF);

  const Symbol *sym = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("simple_function"), eSymbolTypeAny);
  ASSERT_NE(nullptr, sym);

  std::unique_ptr<UnwindPlan> plan_up = cfi.GetUnwindPlan(sym->GetAddress());
  // Should succeed with valid CIE and FDE
  ASSERT_NE(nullptr, plan_up);
  EXPECT_GE(plan_up->GetRowCount(), 1);
}
