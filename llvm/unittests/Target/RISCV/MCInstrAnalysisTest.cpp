//===- MCInstrAnalysisTest.cpp - RISCVMCInstrAnalysis unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrAnalysis.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

class InstrAnalysisTest : public testing::TestWithParam<const char *> {
protected:
  std::unique_ptr<const MCInstrInfo> Info;
  std::unique_ptr<const MCInstrAnalysis> Analysis;

  static void SetUpTestSuite() {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
  }

  InstrAnalysisTest() {
    std::string Error;
    Triple TT(Triple::normalize(GetParam()));
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    Info = std::unique_ptr<const MCInstrInfo>(TheTarget->createMCInstrInfo());
    Analysis = std::unique_ptr<const MCInstrAnalysis>(
        TheTarget->createMCInstrAnalysis(Info.get()));
  }
};

} // namespace

static MCInst jal(unsigned RD) {
  return MCInstBuilder(RISCV::JAL).addReg(RD).addImm(16);
}

static MCInst jalr(unsigned RD, unsigned RS1 = RISCV::X10) {
  return MCInstBuilder(RISCV::JALR).addReg(RD).addReg(RS1).addImm(16);
}

static MCInst cjr(unsigned RS1) {
  return MCInstBuilder(RISCV::C_JR).addReg(RS1);
}

static MCInst cj() { return MCInstBuilder(RISCV::C_J).addImm(16); }
static MCInst cjal() { return MCInstBuilder(RISCV::C_JAL).addImm(16); }

static MCInst cjalr(unsigned RS1) {
  return MCInstBuilder(RISCV::C_JALR).addReg(RS1);
}

static MCInst beq() {
  return MCInstBuilder(RISCV::BEQ)
      .addReg(RISCV::X0)
      .addReg(RISCV::X1)
      .addImm(32);
}

static MCInst cbeqz() {
  return MCInstBuilder(RISCV::C_BEQZ).addReg(RISCV::X1).addImm(32);
}

TEST_P(InstrAnalysisTest, IsTerminator) {
  EXPECT_TRUE(Analysis->isTerminator(beq()));
  EXPECT_TRUE(Analysis->isTerminator(cbeqz()));
  EXPECT_TRUE(Analysis->isTerminator(jal(RISCV::X0)));
  EXPECT_FALSE(Analysis->isTerminator(jal(RISCV::X5)));
  EXPECT_TRUE(Analysis->isTerminator(jalr(RISCV::X0)));
  EXPECT_FALSE(Analysis->isTerminator(jalr(RISCV::X5)));
  EXPECT_TRUE(Analysis->isTerminator(cj()));
  EXPECT_FALSE(Analysis->isTerminator(cjal()));
}

TEST_P(InstrAnalysisTest, IsCall) {
  EXPECT_FALSE(Analysis->isCall(beq()));
  EXPECT_FALSE(Analysis->isCall(cbeqz()));
  EXPECT_FALSE(Analysis->isCall(jal(RISCV::X0)));
  EXPECT_TRUE(Analysis->isCall(jal(RISCV::X1)));
  EXPECT_TRUE(Analysis->isCall(jalr(RISCV::X1, RISCV::X1)));
  EXPECT_FALSE(Analysis->isCall(jalr(RISCV::X0, RISCV::X5)));
  EXPECT_FALSE(Analysis->isCall(cj()));
  EXPECT_FALSE(Analysis->isCall(cjr(RISCV::X5)));
  EXPECT_TRUE(Analysis->isCall(cjal()));
  EXPECT_TRUE(Analysis->isCall(cjalr(RISCV::X5)));
}

TEST_P(InstrAnalysisTest, IsReturn) {
  EXPECT_FALSE(Analysis->isReturn(beq()));
  EXPECT_FALSE(Analysis->isReturn(cbeqz()));
  EXPECT_FALSE(Analysis->isReturn(jal(RISCV::X0)));
  EXPECT_TRUE(Analysis->isReturn(jalr(RISCV::X0, RISCV::X1)));
  EXPECT_FALSE(Analysis->isReturn(jalr(RISCV::X1, RISCV::X1)));
  EXPECT_TRUE(Analysis->isReturn(jalr(RISCV::X0, RISCV::X5)));
  EXPECT_FALSE(Analysis->isReturn(cj()));
  EXPECT_TRUE(Analysis->isReturn(cjr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isReturn(cjr(RISCV::X2)));
  EXPECT_TRUE(Analysis->isReturn(cjr(RISCV::X5)));
  EXPECT_FALSE(Analysis->isReturn(cjal()));
  EXPECT_FALSE(Analysis->isReturn(cjalr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isReturn(cjalr(RISCV::X5)));
}

TEST_P(InstrAnalysisTest, IsBranch) {
  EXPECT_TRUE(Analysis->isBranch(beq()));
  EXPECT_TRUE(Analysis->isBranch(cbeqz()));
  EXPECT_TRUE(Analysis->isBranch(jal(RISCV::X0)));
  EXPECT_FALSE(Analysis->isBranch(jal(RISCV::X1)));
  EXPECT_FALSE(Analysis->isBranch(jal(RISCV::X5)));
  EXPECT_TRUE(Analysis->isBranch(jalr(RISCV::X0)));
  EXPECT_FALSE(Analysis->isBranch(jalr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isBranch(jalr(RISCV::X5)));
  EXPECT_FALSE(Analysis->isBranch(jalr(RISCV::X0, RISCV::X1)));
  EXPECT_FALSE(Analysis->isBranch(jalr(RISCV::X0, RISCV::X5)));
  EXPECT_TRUE(Analysis->isBranch(cj()));
  EXPECT_TRUE(Analysis->isBranch(cjr(RISCV::X2)));
  EXPECT_FALSE(Analysis->isBranch(cjr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isBranch(cjr(RISCV::X5)));
  EXPECT_FALSE(Analysis->isBranch(cjal()));
  EXPECT_FALSE(Analysis->isBranch(cjalr(RISCV::X6)));
  EXPECT_FALSE(Analysis->isBranch(cjalr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isBranch(cjalr(RISCV::X5)));
}

TEST_P(InstrAnalysisTest, IsUnconditionalBranch) {
  EXPECT_FALSE(Analysis->isUnconditionalBranch(beq()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(cbeqz()));
  EXPECT_TRUE(Analysis->isUnconditionalBranch(jal(RISCV::X0)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(jal(RISCV::X1)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(jal(RISCV::X5)));
  EXPECT_TRUE(Analysis->isUnconditionalBranch(jalr(RISCV::X0)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(jalr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(jalr(RISCV::X5)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(jalr(RISCV::X0, RISCV::X1)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(jalr(RISCV::X0, RISCV::X5)));
  EXPECT_TRUE(Analysis->isUnconditionalBranch(cj()));
  EXPECT_TRUE(Analysis->isUnconditionalBranch(cjr(RISCV::X2)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(cjr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(cjr(RISCV::X5)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(cjal()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(cjalr(RISCV::X6)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(cjalr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(cjalr(RISCV::X5)));
}

TEST_P(InstrAnalysisTest, IsIndirectBranch) {
  EXPECT_FALSE(Analysis->isIndirectBranch(beq()));
  EXPECT_FALSE(Analysis->isIndirectBranch(cbeqz()));
  EXPECT_FALSE(Analysis->isIndirectBranch(jal(RISCV::X0)));
  EXPECT_FALSE(Analysis->isIndirectBranch(jal(RISCV::X1)));
  EXPECT_TRUE(Analysis->isIndirectBranch(jalr(RISCV::X0)));
  EXPECT_FALSE(Analysis->isIndirectBranch(jalr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isIndirectBranch(cj()));
  EXPECT_TRUE(Analysis->isIndirectBranch(cjr(RISCV::X10)));
  EXPECT_FALSE(Analysis->isIndirectBranch(cjr(RISCV::X1)));
  EXPECT_FALSE(Analysis->isIndirectBranch(cjr(RISCV::X5)));
  EXPECT_FALSE(Analysis->isIndirectBranch(cjal()));
  EXPECT_FALSE(Analysis->isIndirectBranch(cjalr(RISCV::X5)));
}

INSTANTIATE_TEST_SUITE_P(RV32And64, InstrAnalysisTest,
                         testing::Values("riscv32", "riscv64"));
