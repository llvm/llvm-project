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
#include "llvm/IR/DebugInfoMetadata.h"
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
  std::unique_ptr<Module> M;

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
    M = std::make_unique<Module>("Module", *Ctx);
    M->setDataLayout(TM->createDataLayout());
    auto *FType = FunctionType::get(Type::getVoidTy(*Ctx), false);
    auto *F = Function::Create(FType, GlobalValue::ExternalLinkage, "Test", *M);
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

TEST_P(RISCVInstrInfoTest, IsCopyInstrImpl) {
  const RISCVInstrInfo *TII = ST->getInstrInfo();
  DebugLoc DL;

  // ADDI.

  MachineInstr *MI1 = BuildMI(*MF, DL, TII->get(RISCV::ADDI), RISCV::X1)
                          .addReg(RISCV::X2)
                          .addImm(-128)
                          .getInstr();
  auto MI1Res = TII->isCopyInstrImpl(*MI1);
  EXPECT_FALSE(MI1Res.has_value());

  MachineInstr *MI2 = BuildMI(*MF, DL, TII->get(RISCV::ADDI), RISCV::X1)
                          .addReg(RISCV::X2)
                          .addImm(0)
                          .getInstr();
  auto MI2Res = TII->isCopyInstrImpl(*MI2);
  ASSERT_TRUE(MI2Res.has_value());
  EXPECT_EQ(MI2Res->Destination->getReg(), RISCV::X1);
  EXPECT_EQ(MI2Res->Source->getReg(), RISCV::X2);

  // Partial coverage of FSGNJ_* instructions.

  MachineInstr *MI3 = BuildMI(*MF, DL, TII->get(RISCV::FSGNJ_D), RISCV::F1_D)
                          .addReg(RISCV::F2_D)
                          .addReg(RISCV::F1_D)
                          .getInstr();
  auto MI3Res = TII->isCopyInstrImpl(*MI3);
  EXPECT_FALSE(MI3Res.has_value());

  MachineInstr *MI4 = BuildMI(*MF, DL, TII->get(RISCV::FSGNJ_D), RISCV::F1_D)
                          .addReg(RISCV::F2_D)
                          .addReg(RISCV::F2_D)
                          .getInstr();
  auto MI4Res = TII->isCopyInstrImpl(*MI4);
  ASSERT_TRUE(MI4Res.has_value());
  EXPECT_EQ(MI4Res->Destination->getReg(), RISCV::F1_D);
  EXPECT_EQ(MI4Res->Source->getReg(), RISCV::F2_D);

  // ADD. TODO: Should return true for add reg, x0 and add x0, reg.
  MachineInstr *MI5 = BuildMI(*MF, DL, TII->get(RISCV::ADD), RISCV::X1)
                          .addReg(RISCV::X2)
                          .addReg(RISCV::X3)
                          .getInstr();
  auto MI5Res = TII->isCopyInstrImpl(*MI5);
  EXPECT_FALSE(MI5Res.has_value());

  MachineInstr *MI6 = BuildMI(*MF, DL, TII->get(RISCV::ADD), RISCV::X1)
                          .addReg(RISCV::X0)
                          .addReg(RISCV::X2)
                          .getInstr();
  auto MI6Res = TII->isCopyInstrImpl(*MI6);
  EXPECT_FALSE(MI6Res.has_value());

  MachineInstr *MI7 = BuildMI(*MF, DL, TII->get(RISCV::ADD), RISCV::X1)
                          .addReg(RISCV::X2)
                          .addReg(RISCV::X0)
                          .getInstr();
  auto MI7Res = TII->isCopyInstrImpl(*MI7);
  EXPECT_FALSE(MI7Res.has_value());
}

TEST_P(RISCVInstrInfoTest, GetMemOperandsWithOffsetWidth) {
  const RISCVInstrInfo *TII = ST->getInstrInfo();
  const TargetRegisterInfo *TRI = ST->getRegisterInfo();
  DebugLoc DL;

  SmallVector<const MachineOperand *> BaseOps;
  unsigned Width;
  int64_t Offset;
  bool OffsetIsScalable;

  auto MMO = MF->getMachineMemOperand(MachinePointerInfo(),
                                      MachineMemOperand::MOLoad, 1, Align(1));
  MachineInstr *MI = BuildMI(*MF, DL, TII->get(RISCV::LB), RISCV::X1)
                         .addReg(RISCV::X2)
                         .addImm(-128)
                         .addMemOperand(MMO)
                         .getInstr();
  bool Res = TII->getMemOperandsWithOffsetWidth(*MI, BaseOps, Offset,
                                                OffsetIsScalable, Width, TRI);
  ASSERT_TRUE(Res);
  ASSERT_EQ(BaseOps.size(), 1u);
  ASSERT_TRUE(BaseOps.front()->isReg());
  EXPECT_EQ(BaseOps.front()->getReg(), RISCV::X2);
  EXPECT_EQ(Offset, -128);
  EXPECT_FALSE(OffsetIsScalable);
  EXPECT_EQ(Width, 1u);

  BaseOps.clear();
  MMO = MF->getMachineMemOperand(MachinePointerInfo(),
                                 MachineMemOperand::MOStore, 4, Align(4));
  MI = BuildMI(*MF, DL, TII->get(RISCV::FSW))
           .addReg(RISCV::F3_F)
           .addReg(RISCV::X3)
           .addImm(36)
           .addMemOperand(MMO);
  Res = TII->getMemOperandsWithOffsetWidth(*MI, BaseOps, Offset,
                                           OffsetIsScalable, Width, TRI);
  ASSERT_TRUE(Res);
  ASSERT_EQ(BaseOps.size(), 1u);
  ASSERT_TRUE(BaseOps.front()->isReg());
  EXPECT_EQ(BaseOps.front()->getReg(), RISCV::X3);
  EXPECT_EQ(Offset, 36);
  EXPECT_FALSE(OffsetIsScalable);
  EXPECT_EQ(Width, 4u);

  BaseOps.clear();
  MMO = MF->getMachineMemOperand(MachinePointerInfo(),
                                 MachineMemOperand::MOStore, 16, Align(16));
  MI = BuildMI(*MF, DL, TII->get(RISCV::PseudoVLE32_V_M1), RISCV::V8)
           .addReg(RISCV::X3)
           .addMemOperand(MMO);
  Res = TII->getMemOperandsWithOffsetWidth(*MI, BaseOps, Offset,
                                           OffsetIsScalable, Width, TRI);
  ASSERT_FALSE(Res); // Vector loads/stored are not handled for now.

  BaseOps.clear();
  MI = BuildMI(*MF, DL, TII->get(RISCV::ADDI), RISCV::X4)
           .addReg(RISCV::X5)
           .addImm(16);
  Res = TII->getMemOperandsWithOffsetWidth(*MI, BaseOps, Offset,
                                           OffsetIsScalable, Width, TRI);

  BaseOps.clear();
  MMO = MF->getMachineMemOperand(MachinePointerInfo(),
                                 MachineMemOperand::MOStore, 4, Align(4));
  MI = BuildMI(*MF, DL, TII->get(RISCV::SW))
           .addReg(RISCV::X3)
           .addFrameIndex(2)
           .addImm(4)
           .addMemOperand(MMO);
  Res = TII->getMemOperandsWithOffsetWidth(*MI, BaseOps, Offset,
                                           OffsetIsScalable, Width, TRI);
  ASSERT_TRUE(Res);
  ASSERT_EQ(BaseOps.size(), 1u);
  ASSERT_TRUE(BaseOps.front()->isFI());
  EXPECT_EQ(BaseOps.front()->getIndex(), 2);
  EXPECT_EQ(Offset, 4);
  EXPECT_FALSE(OffsetIsScalable);
  EXPECT_EQ(Width, 4u);
}

static void expectDIEPrintResult(const DIExpression *Expr, StringRef Expected) {
  std::string Output;
  raw_string_ostream OS(Output);
  Expr->print(OS);
  OS.flush();
  EXPECT_EQ(OS.str(), Expected);
}

TEST_P(RISCVInstrInfoTest, DescribeLoadedValue) {
  const RISCVInstrInfo *TII = ST->getInstrInfo();
  DebugLoc DL;

  MachineBasicBlock *MBB = MF->CreateMachineBasicBlock();
  MF->getProperties().set(MachineFunctionProperties::Property::NoVRegs);

  // Register move.
  auto *MI1 = BuildMI(*MBB, MBB->begin(), DL, TII->get(RISCV::ADDI), RISCV::X1)
                  .addReg(RISCV::X2)
                  .addImm(0)
                  .getInstr();
  EXPECT_FALSE(TII->describeLoadedValue(*MI1, RISCV::X2).has_value());
  std::optional<ParamLoadedValue> MI1Res =
      TII->describeLoadedValue(*MI1, RISCV::X1);
  ASSERT_TRUE(MI1Res.has_value());
  ASSERT_TRUE(MI1Res->first.isReg());
  EXPECT_EQ(MI1Res->first.getReg(), RISCV::X2);
  expectDIEPrintResult(MI1Res->second, "!DIExpression()");

  // Load immediate.
  auto *MI2 = BuildMI(*MBB, MBB->begin(), DL, TII->get(RISCV::ADDI), RISCV::X3)
                  .addReg(RISCV::X0)
                  .addImm(111)
                  .getInstr();
  std::optional<ParamLoadedValue> MI2Res =
      TII->describeLoadedValue(*MI2, RISCV::X3);
  ASSERT_TRUE(MI2Res.has_value());
  ASSERT_TRUE(MI2Res->first.isReg());
  EXPECT_EQ(MI2Res->first.getReg(), RISCV::X0);
  // TODO: Could be a DW_OP_constu if this is recognised as a immediate load
  // rather than just an addi.
  expectDIEPrintResult(MI2Res->second, "!DIExpression(DW_OP_plus_uconst, 111)");

  // Add immediate.
  auto *MI3 = BuildMI(*MBB, MBB->begin(), DL, TII->get(RISCV::ADDI), RISCV::X2)
                  .addReg(RISCV::X3)
                  .addImm(222)
                  .getInstr();
  std::optional<ParamLoadedValue> MI3Res =
      TII->describeLoadedValue(*MI3, RISCV::X2);
  ASSERT_TRUE(MI3Res.has_value());
  ASSERT_TRUE(MI3Res->first.isReg());
  EXPECT_EQ(MI3Res->first.getReg(), RISCV::X3);
  expectDIEPrintResult(MI3Res->second, "!DIExpression(DW_OP_plus_uconst, 222)");

  // Load value from memory.
  // It would be better (more reflective of real-world describeLoadedValue
  // usage) to test using MachinePointerInfo::getFixedStack, but
  // unfortunately it would be overly fiddly to make this work.
  auto MMO = MF->getMachineMemOperand(MachinePointerInfo::getConstantPool(*MF),
                                      MachineMemOperand::MOLoad, 1, Align(1));
  auto *MI4 = BuildMI(*MBB, MBB->begin(), DL, TII->get(RISCV::LB), RISCV::X1)
                  .addReg(RISCV::X2)
                  .addImm(-128)
                  .addMemOperand(MMO)
                  .getInstr();
  std::optional<ParamLoadedValue> MI4Res =
      TII->describeLoadedValue(*MI4, RISCV::X1);
  ASSERT_TRUE(MI4Res.has_value());
  ASSERT_TRUE(MI4Res->first.isReg());
  EXPECT_EQ(MI4Res->first.getReg(), RISCV::X2);
  expectDIEPrintResult(
      MI4Res->second,
      "!DIExpression(DW_OP_constu, 128, DW_OP_minus, DW_OP_deref_size, 1)");

  MF->deleteMachineBasicBlock(MBB);
}

} // namespace

INSTANTIATE_TEST_SUITE_P(RV32And64, RISCVInstrInfoTest,
                         testing::Values("riscv32", "riscv64"));
