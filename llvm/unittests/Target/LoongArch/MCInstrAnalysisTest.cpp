//===- MCInstrAnalysisTest.cpp - LoongArchMCInstrAnalysis unit tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrAnalysis.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
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
    LLVMInitializeLoongArchTargetInfo();
    LLVMInitializeLoongArchTarget();
    LLVMInitializeLoongArchTargetMC();
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

static MCInst beq() {
  return MCInstBuilder(LoongArch::BEQ)
      .addReg(LoongArch::R0)
      .addReg(LoongArch::R1)
      .addImm(32);
}

static MCInst b() { return MCInstBuilder(LoongArch::B).addImm(32); }

static MCInst bl() { return MCInstBuilder(LoongArch::BL).addImm(32); }

static MCInst jirl(unsigned RD, unsigned RJ = LoongArch::R10) {
  return MCInstBuilder(LoongArch::JIRL).addReg(RD).addReg(RJ).addImm(16);
}

TEST_P(InstrAnalysisTest, IsTerminator) {
  EXPECT_TRUE(Analysis->isTerminator(beq()));
  EXPECT_TRUE(Analysis->isTerminator(b()));
  EXPECT_FALSE(Analysis->isTerminator(bl()));
  EXPECT_TRUE(Analysis->isTerminator(jirl(LoongArch::R0)));
  EXPECT_FALSE(Analysis->isTerminator(jirl(LoongArch::R5)));
}

TEST_P(InstrAnalysisTest, IsCall) {
  EXPECT_FALSE(Analysis->isCall(beq()));
  EXPECT_FALSE(Analysis->isCall(b()));
  EXPECT_TRUE(Analysis->isCall(bl()));
  EXPECT_TRUE(Analysis->isCall(jirl(LoongArch::R1)));
  EXPECT_FALSE(Analysis->isCall(jirl(LoongArch::R0)));
}

TEST_P(InstrAnalysisTest, IsReturn) {
  EXPECT_FALSE(Analysis->isReturn(beq()));
  EXPECT_FALSE(Analysis->isReturn(b()));
  EXPECT_FALSE(Analysis->isReturn(bl()));
  EXPECT_TRUE(Analysis->isReturn(jirl(LoongArch::R0, LoongArch::R1)));
  EXPECT_FALSE(Analysis->isReturn(jirl(LoongArch::R0)));
  EXPECT_FALSE(Analysis->isReturn(jirl(LoongArch::R1)));
}

TEST_P(InstrAnalysisTest, IsBranch) {
  EXPECT_TRUE(Analysis->isBranch(beq()));
  EXPECT_TRUE(Analysis->isBranch(b()));
  EXPECT_FALSE(Analysis->isBranch(bl()));
  EXPECT_TRUE(Analysis->isBranch(jirl(LoongArch::R0)));
  EXPECT_FALSE(Analysis->isBranch(jirl(LoongArch::R1)));
  EXPECT_FALSE(Analysis->isBranch(jirl(LoongArch::R0, LoongArch::R1)));
}

TEST_P(InstrAnalysisTest, IsConditionalBranch) {
  EXPECT_TRUE(Analysis->isConditionalBranch(beq()));
  EXPECT_FALSE(Analysis->isConditionalBranch(b()));
  EXPECT_FALSE(Analysis->isConditionalBranch(bl()));
}

TEST_P(InstrAnalysisTest, IsUnconditionalBranch) {
  EXPECT_FALSE(Analysis->isUnconditionalBranch(beq()));
  EXPECT_TRUE(Analysis->isUnconditionalBranch(b()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(bl()));
  EXPECT_TRUE(Analysis->isUnconditionalBranch(jirl(LoongArch::R0)));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(jirl(LoongArch::R1)));
  EXPECT_FALSE(
      Analysis->isUnconditionalBranch(jirl(LoongArch::R0, LoongArch::R1)));
}

TEST_P(InstrAnalysisTest, IsIndirectBranch) {
  EXPECT_FALSE(Analysis->isIndirectBranch(beq()));
  EXPECT_FALSE(Analysis->isIndirectBranch(b()));
  EXPECT_FALSE(Analysis->isIndirectBranch(bl()));
  EXPECT_TRUE(Analysis->isIndirectBranch(jirl(LoongArch::R0)));
  EXPECT_FALSE(Analysis->isIndirectBranch(jirl(LoongArch::R1)));
  EXPECT_FALSE(Analysis->isIndirectBranch(jirl(LoongArch::R0, LoongArch::R1)));
}

INSTANTIATE_TEST_SUITE_P(LA32And64, InstrAnalysisTest,
                         testing::Values("loongarch32", "loongarch64"));
