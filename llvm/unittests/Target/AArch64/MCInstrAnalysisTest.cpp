//===- MCInstrAnalysisTest.cpp - AArch64MCInstrAnalysis unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrAnalysis.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "Utils/AArch64BaseInfo.h"
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
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
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
  return MCInstBuilder(AArch64::Bcc).addImm(AArch64CC::EQ).addReg(AArch64::X0);
}
static MCInst tbz(unsigned Rt = AArch64::X0, unsigned Imm = 0,
                  unsigned Label = 32) {
  return MCInstBuilder(AArch64::TBZX).addReg(Rt).addImm(Imm).addImm(Label);
}
static MCInst cbz(unsigned Rt = AArch64::X0, unsigned Label = 32) {
  return MCInstBuilder(AArch64::CBZX).addReg(Rt).addImm(Label);
}
static MCInst b() { return MCInstBuilder(AArch64::B).addImm(32); }
static MCInst bl() { return MCInstBuilder(AArch64::BL).addImm(32); }
static MCInst br(unsigned Rn = AArch64::X0) {
  return MCInstBuilder(AArch64::BR).addReg(Rn);
}
static MCInst blr(unsigned Rn = AArch64::X0) {
  return MCInstBuilder(AArch64::BLR).addReg(Rn);
}
static MCInst ret(unsigned Rn = AArch64::LR) {
  return MCInstBuilder(AArch64::RET).addReg(Rn);
}
static MCInst retaa() { return MCInstBuilder(AArch64::RETAA); }
static MCInst eret() { return MCInstBuilder(AArch64::ERET); }
static MCInst hlt() { return MCInstBuilder(AArch64::HLT); }
static MCInst brk() { return MCInstBuilder(AArch64::BRK); }
static MCInst svc() { return MCInstBuilder(AArch64::SVC); }
static MCInst hvc() { return MCInstBuilder(AArch64::HVC); }
static MCInst smc() { return MCInstBuilder(AArch64::SMC); }

TEST_P(InstrAnalysisTest, IsTerminator) {
  EXPECT_TRUE(Analysis->isTerminator(beq()));
  EXPECT_TRUE(Analysis->isTerminator(tbz()));
  EXPECT_TRUE(Analysis->isTerminator(cbz()));
  EXPECT_TRUE(Analysis->isTerminator(b()));
  EXPECT_FALSE(Analysis->isTerminator(bl()));
  EXPECT_FALSE(Analysis->isTerminator(blr()));
  EXPECT_TRUE(Analysis->isTerminator(br()));
  EXPECT_TRUE(Analysis->isTerminator(ret()));
  EXPECT_TRUE(Analysis->isTerminator(retaa()));
  EXPECT_TRUE(Analysis->isTerminator(eret()));
  EXPECT_FALSE(Analysis->isTerminator(hlt()));
  EXPECT_FALSE(Analysis->isTerminator(brk()));
  EXPECT_FALSE(Analysis->isTerminator(svc()));
  EXPECT_FALSE(Analysis->isTerminator(hvc()));
  EXPECT_FALSE(Analysis->isTerminator(smc()));
}

TEST_P(InstrAnalysisTest, IsBarrier) {
  EXPECT_FALSE(Analysis->isBarrier(beq()));
  EXPECT_FALSE(Analysis->isBarrier(tbz()));
  EXPECT_FALSE(Analysis->isBarrier(cbz()));
  EXPECT_TRUE(Analysis->isBarrier(b()));
  EXPECT_FALSE(Analysis->isBarrier(bl()));
  EXPECT_FALSE(Analysis->isBarrier(blr()));
  EXPECT_TRUE(Analysis->isBarrier(br()));
  EXPECT_TRUE(Analysis->isBarrier(ret()));
  EXPECT_TRUE(Analysis->isBarrier(retaa()));
  EXPECT_TRUE(Analysis->isBarrier(eret()));
  EXPECT_FALSE(Analysis->isBarrier(hlt()));
  EXPECT_FALSE(Analysis->isBarrier(brk()));
  EXPECT_FALSE(Analysis->isBarrier(svc()));
  EXPECT_FALSE(Analysis->isBarrier(hvc()));
  EXPECT_FALSE(Analysis->isBarrier(smc()));
}

TEST_P(InstrAnalysisTest, IsCall) {
  EXPECT_FALSE(Analysis->isCall(beq()));
  EXPECT_FALSE(Analysis->isCall(tbz()));
  EXPECT_FALSE(Analysis->isCall(cbz()));
  EXPECT_FALSE(Analysis->isCall(b()));
  EXPECT_TRUE(Analysis->isCall(bl()));
  EXPECT_TRUE(Analysis->isCall(blr()));
  EXPECT_FALSE(Analysis->isCall(br()));
  EXPECT_FALSE(Analysis->isCall(ret()));
  EXPECT_FALSE(Analysis->isCall(retaa()));
  EXPECT_FALSE(Analysis->isCall(eret()));
}

TEST_P(InstrAnalysisTest, IsReturn) {
  EXPECT_FALSE(Analysis->isReturn(beq()));
  EXPECT_FALSE(Analysis->isReturn(tbz()));
  EXPECT_FALSE(Analysis->isReturn(cbz()));
  EXPECT_FALSE(Analysis->isReturn(b()));
  EXPECT_FALSE(Analysis->isReturn(bl()));
  EXPECT_FALSE(Analysis->isReturn(br()));
  EXPECT_FALSE(Analysis->isReturn(blr()));
  EXPECT_FALSE(Analysis->isReturn(br(AArch64::LR)));
  EXPECT_TRUE(Analysis->isReturn(ret()));
  EXPECT_TRUE(Analysis->isReturn(retaa()));
  EXPECT_TRUE(Analysis->isReturn(eret()));
}

TEST_P(InstrAnalysisTest, IsBranch) {
  EXPECT_TRUE(Analysis->isBranch(beq()));
  EXPECT_TRUE(Analysis->isBranch(tbz()));
  EXPECT_TRUE(Analysis->isBranch(cbz()));
  EXPECT_TRUE(Analysis->isBranch(b()));
  EXPECT_FALSE(Analysis->isBranch(bl()));
  EXPECT_FALSE(Analysis->isBranch(blr()));
  EXPECT_TRUE(Analysis->isBranch(br()));
  EXPECT_FALSE(Analysis->isBranch(ret()));
  EXPECT_FALSE(Analysis->isBranch(retaa()));
  EXPECT_FALSE(Analysis->isBranch(eret()));
}

TEST_P(InstrAnalysisTest, IsConditionalBranch) {
  EXPECT_TRUE(Analysis->isConditionalBranch(beq()));
  EXPECT_TRUE(Analysis->isConditionalBranch(tbz()));
  EXPECT_TRUE(Analysis->isConditionalBranch(cbz()));
  EXPECT_FALSE(Analysis->isConditionalBranch(b()));
  EXPECT_FALSE(Analysis->isConditionalBranch(bl()));
  EXPECT_FALSE(Analysis->isConditionalBranch(blr()));
  EXPECT_FALSE(Analysis->isConditionalBranch(ret()));
  EXPECT_FALSE(Analysis->isConditionalBranch(retaa()));
  EXPECT_FALSE(Analysis->isConditionalBranch(eret()));
}

TEST_P(InstrAnalysisTest, IsUnconditionalBranch) {
  EXPECT_FALSE(Analysis->isUnconditionalBranch(beq()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(tbz()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(cbz()));
  EXPECT_TRUE(Analysis->isUnconditionalBranch(b()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(bl()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(blr()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(br()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(ret()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(retaa()));
  EXPECT_FALSE(Analysis->isUnconditionalBranch(eret()));
}

TEST_P(InstrAnalysisTest, IsIndirectBranch) {
  EXPECT_FALSE(Analysis->isIndirectBranch(beq()));
  EXPECT_FALSE(Analysis->isIndirectBranch(tbz()));
  EXPECT_FALSE(Analysis->isIndirectBranch(cbz()));
  EXPECT_FALSE(Analysis->isIndirectBranch(b()));
  EXPECT_FALSE(Analysis->isIndirectBranch(bl()));
  EXPECT_FALSE(Analysis->isIndirectBranch(blr()));
  EXPECT_TRUE(Analysis->isIndirectBranch(br()));
  EXPECT_FALSE(Analysis->isIndirectBranch(ret()));
  EXPECT_FALSE(Analysis->isIndirectBranch(retaa()));
  EXPECT_FALSE(Analysis->isIndirectBranch(eret()));
}

INSTANTIATE_TEST_SUITE_P(AArch64, InstrAnalysisTest,
                         testing::Values("aarch64"));
