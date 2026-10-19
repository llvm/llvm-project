//===- KnownFPClassTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "gtest/gtest.h"

// These tests exercise queries that the `print<gisel-value-tracking-fpclass>`
// pass cannot represent: the signaling-NaN query, and ppcf128 types, which the
// MIR parser cannot spell.

TEST_F(AArch64GISelMITest, TestFPClassFPowPosNeverSNaN) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %exp:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = G_FABS %val
    %fpow:_(s32) = G_FPOW %fabs, %exp
    %copy_fpow:_(s32) = COPY %fpow
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);
  EXPECT_TRUE(Info.isKnownNeverNaN(SrcReg, true));
}

TEST_F(AArch64GISelMITest, TestFPClassFLogPPCF128) {
  StringRef MIRString = R"(
    %val:_(s128) = G_IMPLICIT_DEF
    %fabs:_(s128) = nnan G_FABS %val
    %flog:_(s128) = G_FLOG %fabs
    %copy_flog:_(s128) = COPY %flog
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies.back();
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  // The MIR parser cannot spell a ppcf128 LLT.
  MachineInstr *FLog = MRI->getVRegDef(SrcReg);
  Register ValReg = FLog->getOperand(1).getReg();
  MRI->setType(ValReg, LLT::ppcf128());
  MRI->setType(SrcReg, LLT::ppcf128());

  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcNegative | fcPositive, Known.getKnownFPClasses());
  EXPECT_EQ(std::nullopt, Known.getSignBit());
}

// TODO: The textual MIR parser does not support the ppcf128 LLT, so we have to
// construct these instructions directly as a workaround.
TEST_F(AArch64GISelMITest, TestFPClassPPCF128TruncNoInf) {
  // ppcf128 trunc cannot introduce +-Inf.
  setUp();
  if (!TM)
    GTEST_SKIP();
  LLT Ty = LLT::ppcf128();
  auto X = B.buildUndef(Ty);
  auto NoInf = B.buildInstr(TargetOpcode::G_FADD, {Ty}, {X, X},
                            MachineInstr::MIFlag::FmNoInfs);
  auto Trunc = B.buildInstr(TargetOpcode::G_INTRINSIC_TRUNC, {Ty}, {NoInf});
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(Trunc.getReg(0));
  EXPECT_TRUE(Known.isKnownNeverPosInfinity());
  EXPECT_TRUE(Known.isKnownNeverNegInfinity());
}

TEST_F(AArch64GISelMITest, TestFPClassPPCF128FloorNoInf) {
  // ppcf128 floor may introduce -Inf, but cannot introduce +Inf.
  setUp();
  if (!TM)
    GTEST_SKIP();
  LLT Ty = LLT::ppcf128();
  auto X = B.buildUndef(Ty);
  auto NoInf = B.buildInstr(TargetOpcode::G_FADD, {Ty}, {X, X},
                            MachineInstr::MIFlag::FmNoInfs);
  auto Floor = B.buildInstr(TargetOpcode::G_FFLOOR, {Ty}, {NoInf});
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(Floor.getReg(0));
  EXPECT_TRUE(Known.isKnownNeverPosInfinity());
  EXPECT_FALSE(Known.isKnownNeverNegInfinity());
}

TEST_F(AArch64GISelMITest, TestFPClassPPCF128CeilNoInf) {
  // ppcf128 ceil may introduce +Inf, but cannot introduce -Inf.
  setUp();
  if (!TM)
    GTEST_SKIP();
  LLT Ty = LLT::ppcf128();
  auto X = B.buildUndef(Ty);
  auto NoInf = B.buildInstr(TargetOpcode::G_FADD, {Ty}, {X, X},
                            MachineInstr::MIFlag::FmNoInfs);
  auto Ceil = B.buildInstr(TargetOpcode::G_FCEIL, {Ty}, {NoInf});
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(Ceil.getReg(0));
  EXPECT_FALSE(Known.isKnownNeverPosInfinity());
  EXPECT_TRUE(Known.isKnownNeverNegInfinity());
}
