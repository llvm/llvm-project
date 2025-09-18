//===-- TestRiscvInstEmulation.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"

#include "lldb/Core/AddressRange.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Utility/ArchSpec.h"

#include "Plugins/Disassembler/LLVMC/DisassemblerLLVMC.h"
#include "Plugins/Instruction/RISCV/EmulateInstructionRISCV.h"
#include "Plugins/Process/Utility/lldb-riscv-register-enums.h"
#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class TestRiscvInstEmulation : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

protected:
};

void TestRiscvInstEmulation::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
  EmulateInstructionRISCV::Initialize();
}

void TestRiscvInstEmulation::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
  EmulateInstructionRISCV::Terminate();
}

TEST_F(TestRiscvInstEmulation, TestSimpleRiscvFunction) {
  ArchSpec arch("riscv64-unknown-linux-gnu");
  // Enable compressed instruction support (RVC extension).
  arch.SetFlags(ArchSpec::eRISCV_rvc);
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  // RISC-V function with compressed and uncompressed instructions
  //   0x0000: 1141          addi    sp, sp, -0x10
  //   0x0002: e406          sd      ra, 0x8(sp)
  //   0x0004: e022          sd      s0, 0x0(sp)
  //   0x0006: 0800          addi    s0, sp, 0x10
  //   0x0008: 00000537      lui     a0, 0x0
  //   0x000C: 00050513      mv      a0, a0
  //   0x0010: 00000097      auipc   ra, 0x0
  //   0x0014: 000080e7      jalr    ra <main+0x10>
  //   0x0018: 4501          li      a0, 0x0
  //   0x001A: ff040113      addi    sp, s0, -0x10
  //   0x001E: 60a2          ld      ra, 0x8(sp)
  //   0x0020: 6402          ld      s0, 0x0(sp)
  //   0x0022: 0141          addi    sp, sp, 0x10
  //   0x0024: 8082          ret
  uint8_t data[] = {// 0x0000: 1141          addi sp, sp, -0x10
                    0x41, 0x11,
                    // 0x0002: e406          sd ra, 0x8(sp)
                    0x06, 0xE4,
                    // 0x0004: e022          sd s0, 0x0(sp)
                    0x22, 0xE0,
                    // 0x0006: 0800          addi s0, sp, 0x10
                    0x00, 0x08,
                    // 0x0008: 00000537      lui a0, 0x0
                    0x37, 0x05, 0x00, 0x00,
                    // 0x000C: 00050513      mv a0, a0
                    0x13, 0x05, 0x05, 0x00,
                    // 0x0010: 00000097      auipc ra, 0x0
                    0x97, 0x00, 0x00, 0x00,
                    // 0x0014: 000080e7      jalr ra <main+0x10>
                    0xE7, 0x80, 0x00, 0x00,
                    // 0x0018: 4501          li a0, 0x0
                    0x01, 0x45,
                    // 0x001A: ff040113      addi sp, s0, -0x10
                    0x13, 0x01, 0x04, 0xFF,
                    // 0x001E: 60a2          ld ra, 0x8(sp)
                    0xA2, 0x60,
                    // 0x0020: 6402          ld s0, 0x0(sp)
                    0x02, 0x64,
                    // 0x0022: 0141          addi sp, sp, 0x10
                    0x41, 0x01,
                    // 0x0024: 8082          ret
                    0x82, 0x80};

  // Expected UnwindPlan (prologue only - emulation stops after frame setup):
  // row[0]:    0:  CFA=sp+0   => fp= <same>        ra= <same>
  // row[1]:    2:  CFA=sp+16  => fp= <same>        ra= <same>      (after stack
  // allocation) row[2]:    4:  CFA=sp+16  => fp= <same>        ra=[CFA-8]
  // (after saving ra) row[3]:    6:  CFA=sp+16  => fp=[CFA-16]       ra=[CFA-8]
  // (after saving s0/fp) row[4]:    8:  CFA=s0+0   => fp=[CFA-16] ra=[CFA-8]
  // (after setting frame pointer: s0=sp+16)

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;
  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), unwind_plan));

  // CFA=sp+0 => fp=<same> ra=<same>.
  row = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_riscv);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_riscv, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_ra_riscv, regloc));
  EXPECT_TRUE(regloc.IsSame());

  // CFA=sp+16 => fp=<same> ra=<same>.
  row = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_riscv);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_riscv, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_ra_riscv, regloc));
  EXPECT_TRUE(regloc.IsSame());

  // CFA=sp+16 => fp=<same> ra=[CFA-8].
  row = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_riscv);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_riscv, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_ra_riscv, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=sp+16 => fp=[CFA-16] ra=[CFA-8]
  row = unwind_plan.GetRowForFunctionOffset(6);
  EXPECT_EQ(6, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_riscv);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_riscv, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_ra_riscv, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=s0+0 => fp=[CFA-16] ra=[CFA-8]
  // s0 = sp + 16, so switching CFA to s0 does not change the effective
  // locations.
  row = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_riscv);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_riscv, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_ra_riscv, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());
}
