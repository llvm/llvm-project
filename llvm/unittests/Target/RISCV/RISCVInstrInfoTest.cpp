//===- RISCVInstrInfoTest.cpp - RISCVInstrInfo unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

class RISCVInstrInfoTest : public testing::TestWithParam<const char *> {
protected:
  std::unique_ptr<RISCVTargetMachine> TM;
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<RISCVSubtarget> ST;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MachineFunction> MF;

  static void SetUpTestSuite() {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
  }

  RISCVInstrInfoTest() {
    std::string Error;
    auto TT(Triple::normalize(GetParam()));
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    TargetOptions Options;

    TM.reset(static_cast<RISCVTargetMachine *>(TheTarget->createTargetMachine(
        TT, "generic", "", Options, std::nullopt, std::nullopt,
        CodeGenOptLevel::Default)));

    Ctx = std::make_unique<LLVMContext>();
    Module M("Module", *Ctx);
    M.setDataLayout(TM->createDataLayout());
    auto *FType = FunctionType::get(Type::getVoidTy(*Ctx), false);
    auto *F = Function::Create(FType, GlobalValue::ExternalLinkage, "Test", &M);
    MMI = std::make_unique<MachineModuleInfo>(TM.get());

    ST = std::make_unique<RISCVSubtarget>(
        TM->getTargetTriple(), TM->getTargetCPU(), TM->getTargetCPU(),
        TM->getTargetFeatureString(),
        TM->getTargetTriple().isArch64Bit() ? "lp64" : "ilp32", 0, 0, *TM);

    MF = std::make_unique<MachineFunction>(*F, *TM, *ST, 42, *MMI);
  }
};

TEST_P(RISCVInstrInfoTest, IsAddImmediate) {
  const RISCVInstrInfo *TII = ST->getInstrInfo();
  DebugLoc DL;

  MachineInstr *MI1 = BuildMI(*MF, DL, TII->get(RISCV::ADDI), RISCV::X1)
                          .addReg(RISCV::X2)
                          .addImm(-128)
                          .getInstr();
  auto MI1Res = TII->isAddImmediate(*MI1, RISCV::X1);
  ASSERT_TRUE(MI1Res.has_value());
  EXPECT_EQ(MI1Res->Reg, RISCV::X2);
  EXPECT_EQ(MI1Res->Imm, -128);
  EXPECT_FALSE(TII->isAddImmediate(*MI1, RISCV::X2).has_value());

  MachineInstr *MI2 =
      BuildMI(*MF, DL, TII->get(RISCV::LUI), RISCV::X1).addImm(-128).getInstr();
  EXPECT_FALSE(TII->isAddImmediate(*MI2, RISCV::X1));

  // Check ADDIW isn't treated as isAddImmediate.
  if (ST->is64Bit()) {
    MachineInstr *MI3 = BuildMI(*MF, DL, TII->get(RISCV::ADDIW), RISCV::X1)
                            .addReg(RISCV::X2)
                            .addImm(-128)
                            .getInstr();
    EXPECT_FALSE(TII->isAddImmediate(*MI3, RISCV::X1));
  }
}

} // namespace

INSTANTIATE_TEST_SUITE_P(RV32And64, RISCVInstrInfoTest,
                         testing::Values("riscv32", "riscv64"));
