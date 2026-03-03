//===- MemOpAddrModeTest.cpp - Test memory operand addressing modes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MemOpAddrModeTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
  }

  void SetUp() override {
    Triple TT("aarch64-unknown-linux-gnu");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    if (!T)
      GTEST_SKIP() << Error;

    TargetOptions Options;
    TM.reset(T->createTargetMachine(TT, "generic", "+lse", Options,
                                    std::nullopt, std::nullopt,
                                    CodeGenOptLevel::Default));
    MII = TM->getMCInstrInfo();
  }

  std::unique_ptr<TargetMachine> TM;
  const MCInstrInfo *MII = nullptr;
};

TEST_F(MemOpAddrModeTest, IndexedLoads) {
  // LDRXui should have Indexed mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDRXui);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModeIndexed);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), 1);
  EXPECT_EQ(AArch64::getMemOpOffsetIdx(Desc.TSFlags), 2);
  EXPECT_FALSE(AArch64::isMemOpPrePostIdx(Desc.TSFlags));
}

TEST_F(MemOpAddrModeTest, UnscaledLoads) {
  // LDURXi should have Unscaled mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDURXi);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModeUnscaled);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), 1);
  EXPECT_FALSE(AArch64::isMemOpPrePostIdx(Desc.TSFlags));
}

TEST_F(MemOpAddrModeTest, RegisterOffsetLoads) {
  // LDRXroX should have RegOff mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDRXroX);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModeRegOff);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), 1);
}

TEST_F(MemOpAddrModeTest, PreIndexLoads) {
  // LDRXpre should have PreIdx mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDRXpre);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModePreIdx);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), 2);
  EXPECT_TRUE(AArch64::isMemOpPrePostIdx(Desc.TSFlags));
}

TEST_F(MemOpAddrModeTest, PostIndexLoads) {
  // LDRXpost should have PostIdx mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDRXpost);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModePostIdx);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), 2);
  EXPECT_TRUE(AArch64::isMemOpPrePostIdx(Desc.TSFlags));
}

TEST_F(MemOpAddrModeTest, LiteralLoads) {
  // LDRXl should have Literal mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDRXl);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModeLiteral);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), -1);
}

TEST_F(MemOpAddrModeTest, PairLoads) {
  // LDPXi should have Pair mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDPXi);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModePair);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), 2);
  EXPECT_FALSE(AArch64::isMemOpPrePostIdx(Desc.TSFlags));
}

TEST_F(MemOpAddrModeTest, PairPreIndexLoads) {
  // LDPXpre should have PairPre mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDPXpre);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModePairPre);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), 3);
  EXPECT_TRUE(AArch64::isMemOpPrePostIdx(Desc.TSFlags));
}

TEST_F(MemOpAddrModeTest, SIMDLoadNoIndex) {
  // LD1Twov8b should have NoIdx mode
  const MCInstrDesc &Desc = MII->get(AArch64::LD1Twov8b);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModeNoIdx);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), 1);
  EXPECT_EQ(AArch64::getMemOpOffsetIdx(Desc.TSFlags), -1);
}

TEST_F(MemOpAddrModeTest, SIMDLoadPostIndexReg) {
  // LD1Twov8b_POST should have PostIdxReg mode
  const MCInstrDesc &Desc = MII->get(AArch64::LD1Twov8b_POST);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModePostIdxReg);
  EXPECT_TRUE(AArch64::isMemOpPrePostIdx(Desc.TSFlags));
}

TEST_F(MemOpAddrModeTest, AtomicLoads) {
  // LDADDX should have NoIdx mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDADDX);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModeNoIdx);
}

TEST_F(MemOpAddrModeTest, ExclusiveLoads) {
  // LDXRX should have NoIdx mode
  const MCInstrDesc &Desc = MII->get(AArch64::LDXRX);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModeNoIdx);
}

TEST_F(MemOpAddrModeTest, NonMemoryInstructions) {
  // ADDXri should have None mode
  const MCInstrDesc &Desc = MII->get(AArch64::ADDXri);
  uint64_t Mode = Desc.TSFlags & AArch64::MemOpAddrModeMask;
  EXPECT_EQ(Mode, AArch64::MemOpAddrModeNone);
  EXPECT_EQ(AArch64::getMemOpBaseRegIdx(Desc.TSFlags), -1);
}

} // namespace
