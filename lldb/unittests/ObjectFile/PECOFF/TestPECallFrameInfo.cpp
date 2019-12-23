//===-- TestPECallFrameInfo.cpp ------------------------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/Process/Utility/lldb-x86-register-enums.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/CallFrameInfo.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "llvm/Testing/Support/Error.h"

using namespace lldb_private;
using namespace lldb;

class PECallFrameInfoTest : public testing::Test {
  SubsystemRAII<FileSystem, ObjectFilePECOFF> subsystems;

protected:
  void GetUnwindPlan(addr_t file_addr, UnwindPlan &plan) const;
};

void PECallFrameInfoTest::GetUnwindPlan(addr_t file_addr, UnwindPlan &plan) const {
  llvm::Expected<TestFile> ExpectedFile = TestFile::fromYaml(
      R"(
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
  ExportTable:     
    RelativeVirtualAddress: 0
    Size:            0
  ImportTable:     
    RelativeVirtualAddress: 0
    Size:            0
  ResourceTable:   
    RelativeVirtualAddress: 0
    Size:            0
  ExceptionTable:  
    RelativeVirtualAddress: 12288
    Size:            60
  CertificateTable: 
    RelativeVirtualAddress: 0
    Size:            0
  BaseRelocationTable: 
    RelativeVirtualAddress: 0
    Size:            0
  Debug:           
    RelativeVirtualAddress: 0
    Size:            0
  Architecture:    
    RelativeVirtualAddress: 0
    Size:            0
  GlobalPtr:       
    RelativeVirtualAddress: 0
    Size:            0
  TlsTable:        
    RelativeVirtualAddress: 0
    Size:            0
  LoadConfigTable: 
    RelativeVirtualAddress: 0
    Size:            0
  BoundImport:     
    RelativeVirtualAddress: 0
    Size:            0
  IAT:             
    RelativeVirtualAddress: 0
    Size:            0
  DelayImportDescriptor: 
    RelativeVirtualAddress: 0
    Size:            0
  ClrRuntimeHeader: 
    RelativeVirtualAddress: 0
    Size:            0
header:          
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [ IMAGE_FILE_EXECUTABLE_IMAGE, IMAGE_FILE_LARGE_ADDRESS_AWARE ]
sections:        
  - Name:            .text
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    VirtualAddress:  4096
    VirtualSize:     4096
  - Name:            .rdata
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_READ ]
    VirtualAddress:  8192
    VirtualSize:     68
    SectionData:     010C06000C3208F006E00470036002302105020005540D0000100000001100000020000019400E352F74670028646600213465001A3315015E000EF00CE00AD008C00650


# Unwind info at 0x2000:
# 01 0C 06 00    No chained info, prolog size = 0xC, unwind codes size is 6 words, no frame register
# 0C 32          UOP_AllocSmall(2) 3 * 8 + 8 bytes, offset in prolog is 0xC
# 08 F0          UOP_PushNonVol(0) R15(0xF), offset in prolog is 8
# 06 E0          UOP_PushNonVol(0) R14(0xE), offset in prolog is 6
# 04 70          UOP_PushNonVol(0) RDI(7), offset in prolog is 4
# 03 60          UOP_PushNonVol(0) RSI(6), offset in prolog is 3
# 02 30          UOP_PushNonVol(0) RBX(3), offset in prolog is 2
# Corresponding prolog:
# 00    push    rbx
# 02    push    rsi
# 03    push    rdi
# 04    push    r14
# 06    push    r15
# 08    sub     rsp, 20h

# Unwind info at 0x2010:
# 21 05 02 00    Has chained info, prolog size = 5, unwind codes size is 2 words, no frame register
# 05 54 0D 00    UOP_SaveNonVol(4) RBP(5) to RSP + 0xD * 8, offset in prolog is 5
# Chained runtime function:
# 00 10 00 00    Start address is 0x1000
# 00 11 00 00    End address is 0x1100
# 00 20 00 00    Unwind info RVA is 0x2000
# Corresponding prolog:
# 00    mov     [rsp+68h], rbp

# Unwind info at 0x2024:
# 19 40 0E 35    No chained info, prolog size = 0x40, unwind codes size is 0xE words, frame register is RBP, frame register offset is RSP + 3 * 16
# 2F 74 67 00    UOP_SaveNonVol(4) RDI(7) to RSP + 0x67 * 8, offset in prolog is 0x2F
# 28 64 66 00    UOP_SaveNonVol(4) RSI(6) to RSP + 0x66 * 8, offset in prolog is 0x28
# 21 34 65 00    UOP_SaveNonVol(4) RBX(3) to RSP + 0x65 * 8, offset in prolog is 0x21
# 1A 33          UOP_SetFPReg(3), offset in prolog is 0x1A
# 15 01 5E 00    UOP_AllocLarge(1) 0x5E * 8 bytes, offset in prolog is 0x15
# 0E F0          UOP_PushNonVol(0) R15(0xF), offset in prolog is 0xE
# 0C E0          UOP_PushNonVol(0) R14(0xE), offset in prolog is 0xC
# 0A D0          UOP_PushNonVol(0) R13(0xD), offset in prolog is 0xA
# 08 C0          UOP_PushNonVol(0) R12(0xC), offset in prolog is 8
# 06 50          UOP_PushNonVol(0) RBP(5), offset in prolog is 6
# Corresponding prolog:
# 00    mov     [rsp+8], rcx
# 05    push    rbp
# 06    push    r12
# 08    push    r13
# 0A    push    r14
# 0C    push    r15
# 0E    sub     rsp, 2F0h
# 15    lea     rbp, [rsp+30h]
# 1A    mov     [rbp+2F8h], rbx
# 21    mov     [rbp+300h], rsi
# 28    mov     [rbp+308h], rdi

  - Name:            .pdata
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_READ ]
    VirtualAddress:  12288
    VirtualSize:     60
    SectionData:     000000000000000000000000000000000000000000000000001000000011000000200000001100000012000010200000001200000013000024200000

# 00 00 00 00
# 00 00 00 00    Test correct processing of empty runtime functions at begin
# 00 00 00 00

# 00 00 00 00
# 00 00 00 00    Test correct processing of empty runtime functions at begin
# 00 00 00 00

# 00 10 00 00    Start address is 0x1000
# 00 11 00 00    End address is 0x1100
# 00 20 00 00    Unwind info RVA is 0x2000

# 00 11 00 00    Start address is 0x1100
# 00 12 00 00    End address is 0x1200
# 10 20 00 00    Unwind info RVA is 0x2010

# 00 12 00 00    Start address is 0x1200
# 00 13 00 00    End address is 0x1300
# 24 20 00 00    Unwind info RVA is 0x2024

symbols:         []
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  ModuleSP module_sp = std::make_shared<Module>(ModuleSpec(FileSpec(ExpectedFile->name())));
  ObjectFile *object_file = module_sp->GetObjectFile();
  ASSERT_NE(object_file, nullptr);

  std::unique_ptr<CallFrameInfo> cfi = object_file->CreateCallFrameInfo();
  ASSERT_NE(cfi.get(), nullptr);

  SectionList *sect_list = object_file->GetSectionList();
  ASSERT_NE(sect_list, nullptr);

  EXPECT_TRUE(cfi->GetUnwindPlan(Address(file_addr, sect_list), plan));
}

TEST_F(PECallFrameInfoTest, Basic_eh) {
  UnwindPlan plan(eRegisterKindLLDB);
  GetUnwindPlan(0x1001080, plan);
  EXPECT_EQ(plan.GetRowCount(), 7);

  UnwindPlan::Row row;
  row.SetOffset(0);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 8);
  row.SetRegisterLocationToIsCFAPlusOffset(lldb_rsp_x86_64, 0, true);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rip_x86_64, -8, true);
  EXPECT_EQ(*plan.GetRowAtIndex(0), row);

  row.SetOffset(2);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x10);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rbx_x86_64, -0x10, true);
  EXPECT_EQ(*plan.GetRowAtIndex(1), row);

  row.SetOffset(3);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x18);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rsi_x86_64, -0x18, true);
  EXPECT_EQ(*plan.GetRowAtIndex(2), row);

  row.SetOffset(4);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x20);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rdi_x86_64, -0x20, true);
  EXPECT_EQ(*plan.GetRowAtIndex(3), row);

  row.SetOffset(6);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x28);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_r14_x86_64, -0x28, true);
  EXPECT_EQ(*plan.GetRowAtIndex(4), row);

  row.SetOffset(8);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x30);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_r15_x86_64, -0x30, true);
  EXPECT_EQ(*plan.GetRowAtIndex(5), row);

  row.SetOffset(0xC);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x50);
  EXPECT_EQ(*plan.GetRowAtIndex(6), row);
}

TEST_F(PECallFrameInfoTest, Chained_eh) {
  UnwindPlan plan(eRegisterKindLLDB);
  GetUnwindPlan(0x1001180, plan);
  EXPECT_EQ(plan.GetRowCount(), 2);

  UnwindPlan::Row row;
  row.SetOffset(0);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x50);
  row.SetRegisterLocationToIsCFAPlusOffset(lldb_rsp_x86_64, 0, true);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rip_x86_64, -8, true);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rbx_x86_64, -0x10, true);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rsi_x86_64, -0x18, true);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rdi_x86_64, -0x20, true);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_r14_x86_64, -0x28, true);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_r15_x86_64, -0x30, true);
  EXPECT_EQ(*plan.GetRowAtIndex(0), row);

  row.SetOffset(5);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rbp_x86_64, 0x18, true);
  EXPECT_EQ(*plan.GetRowAtIndex(1), row);
}

TEST_F(PECallFrameInfoTest, Frame_reg_eh) {
  UnwindPlan plan(eRegisterKindLLDB);
  GetUnwindPlan(0x1001280, plan);
  EXPECT_EQ(plan.GetRowCount(), 11);

  UnwindPlan::Row row;
  row.SetOffset(0);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 8);
  row.SetRegisterLocationToIsCFAPlusOffset(lldb_rsp_x86_64, 0, true);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rip_x86_64, -8, true);
  EXPECT_EQ(*plan.GetRowAtIndex(0), row);

  row.SetOffset(6);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x10);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rbp_x86_64, -0x10, true);
  EXPECT_EQ(*plan.GetRowAtIndex(1), row);

  row.SetOffset(8);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x18);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_r12_x86_64, -0x18, true);
  EXPECT_EQ(*plan.GetRowAtIndex(2), row);

  row.SetOffset(0xA);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x20);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_r13_x86_64, -0x20, true);
  EXPECT_EQ(*plan.GetRowAtIndex(3), row);

  row.SetOffset(0xC);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x28);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_r14_x86_64, -0x28, true);
  EXPECT_EQ(*plan.GetRowAtIndex(4), row);

  row.SetOffset(0xE);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x30);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_r15_x86_64, -0x30, true);
  EXPECT_EQ(*plan.GetRowAtIndex(5), row);

  row.SetOffset(0x15);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rsp_x86_64, 0x320);
  EXPECT_EQ(*plan.GetRowAtIndex(6), row);

  row.SetOffset(0x1A);
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_rbp_x86_64, 0x2F0);
  EXPECT_EQ(*plan.GetRowAtIndex(7), row);

  row.SetOffset(0x21);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rbx_x86_64, 8, true);
  EXPECT_EQ(*plan.GetRowAtIndex(8), row);

  row.SetOffset(0x28);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rsi_x86_64, 0x10, true);
  EXPECT_EQ(*plan.GetRowAtIndex(9), row);

  row.SetOffset(0x2F);
  row.SetRegisterLocationToAtCFAPlusOffset(lldb_rdi_x86_64, 0x18, true);
  EXPECT_EQ(*plan.GetRowAtIndex(10), row);
}
