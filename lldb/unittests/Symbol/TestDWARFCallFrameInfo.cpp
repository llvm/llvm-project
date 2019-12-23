//===-- TestDWARFCallFrameInfo.cpp ------------------------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

  auto module_sp =
      std::make_shared<Module>(ModuleSpec(FileSpec(ExpectedFile->name())));
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

  UnwindPlan plan(eRegisterKindGeneric);
  ASSERT_TRUE(cfi.GetUnwindPlan(sym->GetAddress(), plan));
  ASSERT_EQ(3, plan.GetRowCount());
  EXPECT_EQ(GetExpectedRow0(), *plan.GetRowAtIndex(0));
  EXPECT_EQ(GetExpectedRow1(), *plan.GetRowAtIndex(1));
  EXPECT_EQ(GetExpectedRow2(), *plan.GetRowAtIndex(2));
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
