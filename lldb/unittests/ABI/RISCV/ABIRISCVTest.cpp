//===-- ABIRISCVTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ABI/RISCV/ABISysV_riscv.h"
#include "lldb/Register/Arch/lldb-riscv-dwarf-register-enums.h"
#include "lldb/Register/RegisterDefinition.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"
#include <vector>


namespace dwarf_regs = lldb_private::registers::riscv::dwarf;
using namespace lldb_private;
using namespace lldb;

class ABIRISCVTestFixture : public testing::TestWithParam<llvm::StringRef> {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();
};

void ABIRISCVTestFixture::SetUpTestCase() {
  LLVMInitializeRISCVTargetInfo();
  LLVMInitializeRISCVTargetMC();
  ABISysV_riscv::Initialize();
}

void ABIRISCVTestFixture::TearDownTestCase() {
  ABISysV_riscv::Terminate();
  llvm::llvm_shutdown();
}

static RegisterDefinition MakeRegister(const char *name) {
  RegisterDefinition reg;
  reg.name = ConstString(name);
  reg.set_name = ConstString("GPR");
  return reg;
}

TEST_P(ABIRISCVTestFixture, AugmentRegisterInfo) {
  ABISP abi_sp = ABI::FindPlugin(ProcessSP(), ArchSpec(GetParam()));
  ASSERT_TRUE(abi_sp);

  std::vector<RegisterDefinition> regs{MakeRegister("ra"), MakeRegister("sp"),
                                       MakeRegister("pc")};
  abi_sp->AugmentRegisterInfo(regs);

  ASSERT_EQ(regs.size(), 3U);
  EXPECT_EQ(regs[0].regnum_dwarf, dwarf_regs::gpr_ra);
  EXPECT_EQ(regs[0].regnum_ehframe, dwarf_regs::gpr_ra);
  EXPECT_EQ(regs[1].regnum_dwarf, dwarf_regs::gpr_sp);
  EXPECT_EQ(regs[2].regnum_dwarf, dwarf_regs::gpr_pc);
  EXPECT_EQ(regs[2].regnum_generic,
            static_cast<uint32_t>(LLDB_REGNUM_GENERIC_PC));
}

TEST_P(ABIRISCVTestFixture, AugmentRegisterInfoFloatingPoint) {
  ABISP abi_sp = ABI::FindPlugin(ProcessSP(), ArchSpec(GetParam()));
  ASSERT_TRUE(abi_sp);

  std::vector<RegisterDefinition> regs{
      MakeRegister("ft0"), MakeRegister("fs0"), MakeRegister("fs1"),
      MakeRegister("fa0"), MakeRegister("fs2"), MakeRegister("fs11"),
      MakeRegister("ft11")};
  abi_sp->AugmentRegisterInfo(regs);

  ASSERT_EQ(regs.size(), 7U);
  EXPECT_EQ(regs[0].regnum_dwarf, dwarf_regs::fpr_f0);
  EXPECT_EQ(regs[0].regnum_ehframe, dwarf_regs::fpr_f0);
  EXPECT_EQ(regs[1].regnum_dwarf, dwarf_regs::fpr_f8);
  EXPECT_EQ(regs[2].regnum_dwarf, dwarf_regs::fpr_f9);
  EXPECT_EQ(regs[3].regnum_dwarf, dwarf_regs::fpr_f10);
  EXPECT_EQ(regs[4].regnum_dwarf, dwarf_regs::fpr_f18);
  EXPECT_EQ(regs[5].regnum_dwarf, dwarf_regs::fpr_f27);
  EXPECT_EQ(regs[6].regnum_dwarf, dwarf_regs::fpr_f31);
  EXPECT_EQ(regs[6].regnum_ehframe, dwarf_regs::fpr_f31);
}

INSTANTIATE_TEST_SUITE_P(ABIRISCVTests, ABIRISCVTestFixture,
                         testing::Values("riscv64-unknown-linux-gnu",
                                         "riscv32-unknown-linux-gnu"));
