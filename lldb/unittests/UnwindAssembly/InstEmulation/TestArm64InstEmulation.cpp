//===-- TestArm64InstEmulation.cpp ------------------------------------*- C++
//-*-===//

//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <vector>

#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"
#include "Utility/ARM64_DWARF_Registers.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/UnwindAssembly.h"

#include "Plugins/Disassembler/llvm/DisassemblerLLVMC.h"
#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class TestArm64InstEmulation : public testing::Test {
public:
  //  static void SetUpTestCase() { }

  //  static void TearDownTestCase() { }

  //  virtual void SetUp() override { }

  //  virtual void TearDown() override { }

protected:
};

static void init() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
  EmulateInstructionARM64::Initialize();
}

static void terminate() {
  DisassemblerLLVMC::Terminate();
  EmulateInstructionARM64::Terminate();
}

TEST_F(TestArm64InstEmulation, TestSimpleDarwinFunction) {

  init();

  ArchSpec arch("arm64-apple-ios10", nullptr);
  UnwindAssemblyInstEmulation *engine =
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch));
  ASSERT_NE(engine, nullptr);

  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::RegisterLocation regloc;

  // 'int main() { }' compiled for arm64-apple-ios with clang
  uint8_t data[] = {
      0xfd, 0x7b, 0xbf, 0xa9, // 0xa9bf7bfd :  stp x29, x30, [sp, #-0x10]!
      0xfd, 0x03, 0x00, 0x91, // 0x910003fd :  mov x29, sp
      0xff, 0x43, 0x00, 0xd1, // 0xd10043ff :  sub sp, sp, #0x10

      0xbf, 0x03, 0x00, 0x91, // 0x910003bf :  mov sp, x29
      0xfd, 0x7b, 0xc1, 0xa8, // 0xa8c17bfd :  ldp x29, x30, [sp], #16
      0xc0, 0x03, 0x5f, 0xd6, // 0xd65f03c0 :  ret
  };

  // UnwindPlan we expect:

  // row[0]:    0: CFA=sp +0 =>
  // row[1]:    4: CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[2]:    8: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[2]:   16: CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[3]:   20: CFA=sp +0 => fp= <same> lr= <same>

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), unwind_plan));

  // CFA=sp +0
  row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  // CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::fp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=sp +0 => fp= <same> lr= <same>
  row_sp = unwind_plan.GetRowForFunctionOffset(20);
  EXPECT_EQ(20, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  terminate();
}

TEST_F(TestArm64InstEmulation, TestMediumDarwinFunction) {
  init();

  ArchSpec arch("arm64-apple-ios10", nullptr);
  UnwindAssemblyInstEmulation *engine =
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch));
  ASSERT_NE(engine, nullptr);

  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::RegisterLocation regloc;

  // disassembly of -[NSPlaceholderString initWithBytes:length:encoding:]
  // from Foundation for iOS.
  uint8_t data[] = {
      0xf6, 0x57, 0xbd, 0xa9, // 0:  0xa9bd57f6 stp x22, x21, [sp, #-48]!
      0xf4, 0x4f, 0x01, 0xa9, // 4:  0xa9014ff4 stp x20, x19, [sp, #16]
      0xfd, 0x7b, 0x02, 0xa9, // 8:  0xa9027bfd stp x29, x30, [sp, #32]
      0xfd, 0x83, 0x00, 0x91, // 12: 0x910083fd add x29, sp, #32
      0xff, 0x43, 0x00, 0xd1, // 16: 0xd10043ff sub sp, sp, #16

      // [... function body ...]
      0x1f, 0x20, 0x03, 0xd5, // 20: 0xd503201f nop

      0xbf, 0x83, 0x00, 0xd1, // 24: 0xd10083bf sub sp, x29, #32
      0xfd, 0x7b, 0x42, 0xa9, // 28: 0xa9427bfd ldp x29, x30, [sp, #32]
      0xf4, 0x4f, 0x41, 0xa9, // 32: 0xa9414ff4 ldp x20, x19, [sp, #16]
      0xf6, 0x57, 0xc3, 0xa8, // 36: 0xa8c357f6 ldp x22, x21, [sp], #48
      0x01, 0x16, 0x09, 0x14, // 40: 0x14091601 b   0x18f640524 ; symbol stub
                              // for: CFStringCreateWithBytes
  };

  // UnwindPlan we expect:
  //  0: CFA=sp +0 =>
  //  4: CFA=sp+48 => x21=[CFA-40] x22=[CFA-48]
  //  8: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // 12: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  // 16: CFA=fp+16 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]

  // [... function body ...]

  // 28: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  // 32: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  // 36: CFA=sp+48 => x19= <same> x20= <same> x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  // 40: CFA=sp +0 => x19= <same> x20= <same> x21= <same> x22= <same> fp= <same>
  // lr= <same>

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), unwind_plan));

  // 0: CFA=sp +0 =>
  row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  // 4: CFA=sp+48 => x21=[CFA-40] x22=[CFA-48]
  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_EQ(48, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x21, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-40, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x22, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-48, regloc.GetOffset());

  // 8: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  row_sp = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_EQ(48, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x19, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-24, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x20, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());

  // 12: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(12);
  EXPECT_EQ(12, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_EQ(48, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 16: CFA=fp+16 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::fp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  // 28: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(28);
  EXPECT_EQ(28, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(48, row_sp->GetCFAValue().GetOffset());

  // 32: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  row_sp = unwind_plan.GetRowForFunctionOffset(32);
  EXPECT_EQ(32, row_sp->GetOffset());

  // I'd prefer if these restored registers were cleared entirely instead of set
  // to IsSame...
  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsSame());

  // 36: CFA=sp+48 => x19= <same> x20= <same> x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  row_sp = unwind_plan.GetRowForFunctionOffset(36);
  EXPECT_EQ(36, row_sp->GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x19, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x20, regloc));
  EXPECT_TRUE(regloc.IsSame());

  // 40: CFA=sp +0 => x19= <same> x20= <same> x21= <same> x22= <same> fp= <same>
  // lr= <same>
  row_sp = unwind_plan.GetRowForFunctionOffset(40);
  EXPECT_EQ(40, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x21, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x22, regloc));
  EXPECT_TRUE(regloc.IsSame());

  terminate();
}
