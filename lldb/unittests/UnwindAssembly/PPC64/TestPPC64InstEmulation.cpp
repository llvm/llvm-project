//===-- TestPPC64InstEmulation.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <vector>

#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/UnwindAssembly.h"
#include "lldb/Utility/ArchSpec.h"

#include "Plugins/Disassembler/LLVMC/DisassemblerLLVMC.h"
#include "Plugins/Instruction/PPC64/EmulateInstructionPPC64.h"
#include "Plugins/Process/Utility/lldb-ppc64le-register-enums.h"
#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class TestPPC64InstEmulation : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

  //  virtual void SetUp() override { }
  //  virtual void TearDown() override { }

protected:
};

void TestPPC64InstEmulation::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
  EmulateInstructionPPC64::Initialize();
}

void TestPPC64InstEmulation::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
  EmulateInstructionPPC64::Terminate();
}

TEST_F(TestPPC64InstEmulation, TestSimpleFunction) {
  ArchSpec arch("powerpc64le-linux-gnu");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::RegisterLocation regloc;

  // prologue and epilogue of:
  // int main() {
  //   int i = test();
  //   return i;
  // }
  //
  // compiled with clang -O0 -g
  uint8_t data[] = {
      // prologue
      0x02, 0x10, 0x40, 0x3c, //  0: lis r2, 4098
      0x00, 0x7f, 0x42, 0x38, //  4: addi r2, r2, 32512
      0xa6, 0x02, 0x08, 0x7c, //  8: mflr r0
      0xf8, 0xff, 0xe1, 0xfb, // 12: std r31, -8(r1)
      0x10, 0x00, 0x01, 0xf8, // 16: std r0, 16(r1)
      0x91, 0xff, 0x21, 0xf8, // 20: stdu r1, -112(r1)
      0x78, 0x0b, 0x3f, 0x7c, // 24: mr r31, r1
      0x00, 0x00, 0x60, 0x38, // 28: li r3, 0
      0x64, 0x00, 0x7f, 0x90, // 32: stw r3, 100(r31)

      // epilogue
      0x70, 0x00, 0x21, 0x38, // 36: addi r1, r1, 112
      0x10, 0x00, 0x01, 0xe8, // 40: ld r0, 16(r1)
      0xf8, 0xff, 0xe1, 0xeb, // 44: ld r31, -8(r1)
      0xa6, 0x03, 0x08, 0x7c, // 48: mtlr r0
      0x20, 0x00, 0x80, 0x4e  // 52: blr
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), unwind_plan));

  // 0: CFA=sp+0
  row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r1_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  // 1: CFA=sp+0 => fp=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r1_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_r31_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 2: CFA=sp+0 => fp=[CFA-8] lr=[CFA+16]
  row_sp = unwind_plan.GetRowForFunctionOffset(20);
  EXPECT_EQ(20ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r1_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_lr_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(16, regloc.GetOffset());

  // 3: CFA=sp+112 => fp=[CFA-8] lr=[CFA+16]
  row_sp = unwind_plan.GetRowForFunctionOffset(24);
  EXPECT_EQ(24ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r1_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(112, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_r31_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_lr_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(16, regloc.GetOffset());

  // 4: CFA=r31+112 => fp=[CFA-8] lr=[CFA+16]
  row_sp = unwind_plan.GetRowForFunctionOffset(28);
  EXPECT_EQ(28ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r31_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(112, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_r31_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_lr_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(16, regloc.GetOffset());

  // 5: CFA=sp+0 => fp=[CFA-8] lr=[CFA+16]
  row_sp = unwind_plan.GetRowForFunctionOffset(40);
  EXPECT_EQ(40ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r1_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_r31_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_lr_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(16, regloc.GetOffset());
}

TEST_F(TestPPC64InstEmulation, TestMediumFunction) {
  ArchSpec arch("powerpc64le-linux-gnu");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::RegisterLocation regloc;

  // prologue and epilogue of main() (call-func.c),
  // with several calls and stack variables.
  //
  // compiled with clang -O0 -g
  uint8_t data[] = {
      // prologue
      0xa6, 0x02, 0x08, 0x7c, //  0: mflr r0
      0xf8, 0xff, 0xe1, 0xfb, //  4: std r31, -8(r1)
      0x10, 0x00, 0x01, 0xf8, //  8: std r0, 16(r1)
      0x78, 0x0b, 0x3e, 0x7c, // 12: mr r30, r1
      0xe0, 0x06, 0x20, 0x78, // 16: clrldi r0, r1, 59
      0xa0, 0xfa, 0x00, 0x20, // 20: subfic r0, r0, -1376
      0x6a, 0x01, 0x21, 0x7c, // 24: stdux r1, r1, r0
      0x78, 0x0b, 0x3f, 0x7c, // 28: mr r31, r1

      // epilogue
      0x00, 0x00, 0x21, 0xe8, // 32: ld r1, 0(r1)
      0x20, 0x00, 0x80, 0x4e  // 36: blr
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), unwind_plan));

  // 0: CFA=sp+0
  row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r1_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  // 1: CFA=sp+0 => fp=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r1_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_r31_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 2: CFA=sp+0 => fp=[CFA-8] lr=[CFA+16]
  row_sp = unwind_plan.GetRowForFunctionOffset(12);
  EXPECT_EQ(12ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r1_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(gpr_lr_ppc64le, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(16, regloc.GetOffset());

  // 3: CFA=r30
  row_sp = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r30_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(32);
  EXPECT_EQ(16ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r30_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  // 4: CFA=sp+0
  row_sp = unwind_plan.GetRowForFunctionOffset(36);
  EXPECT_EQ(36ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == gpr_r1_ppc64le);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());
}
