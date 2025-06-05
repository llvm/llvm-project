//===- KnownFPClassTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "gtest/gtest.h"
#include <optional>

TEST_F(AArch64GISelMITest, TestFPClassCstPosZero) {
  StringRef MIRString = "  %3:_(s32) = G_FCONSTANT float 0.0\n"
                        "  %4:_(s32) = COPY %3\n";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  unsigned CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosZero, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassCstNegZero) {
  StringRef MIRString = "  %3:_(s32) = G_FCONSTANT float -0.0\n"
                        "  %4:_(s32) = COPY %3\n";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcNegZero, Known.KnownFPClasses);
  EXPECT_EQ(true, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassUndef) {
  StringRef MIRString = R"(
    %def:_(s32) = G_IMPLICIT_DEF
    %copy_def:_(s32) = COPY %def
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcAllFlags, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassCstVecNegZero) {
  StringRef MIRString = R"(
   %c0:_(s32) = G_FCONSTANT float -0.0
   %c1:_(s32) = G_FCONSTANT float -0.0
   %c2:_(s32) = G_FCONSTANT float -0.0
   %vector:_(<3 x s32>) = G_BUILD_VECTOR %c0, %c1, %c2
   %copy_vector:_(<3 x s32>) = COPY %vector
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcNegZero, Known.KnownFPClasses);
  EXPECT_EQ(true, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectPos0) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %cond:_(s1) = G_LOAD %ptr(p0) :: (load (s1))
    %lhs:_(s32) = G_FCONSTANT float 0.0
    %rhs:_(s32) = G_FCONSTANT float 0.0
    %sel:_(s32) = G_SELECT %cond, %lhs, %rhs
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosZero, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectNeg0) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %cond:_(s1) = G_LOAD %ptr(p0) :: (load (s1))
    %lhs:_(s32) = G_FCONSTANT float -0.0
    %rhs:_(s32) = G_FCONSTANT float -0.0
    %sel:_(s32) = G_SELECT %cond, %lhs, %rhs
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcNegZero, Known.KnownFPClasses);
  EXPECT_EQ(true, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectPosOrNeg0) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %cond:_(s1) = G_LOAD %ptr(p0) :: (load (s1))
    %lhs:_(s32) = G_FCONSTANT float -0.0
    %rhs:_(s32) = G_FCONSTANT float 0.0
    %sel:_(s32) = G_SELECT %cond, %lhs, %rhs
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcZero, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectPosInf) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %cond:_(s1) = G_LOAD %ptr(p0) :: (load (s1))
    %lhs:_(s32) = G_FCONSTANT float 0x7FF0000000000000
    %rhs:_(s32) = G_FCONSTANT float 0x7FF0000000000000
    %sel:_(s32) = G_SELECT %cond, %lhs, %rhs
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosInf, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectNegInf) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %cond:_(s1) = G_LOAD %ptr(p0) :: (load (s1))
    %lhs:_(s32) = G_FCONSTANT float 0xFFF0000000000000
    %rhs:_(s32) = G_FCONSTANT float 0xFFF0000000000000
    %sel:_(s32) = G_SELECT %cond, %lhs, %rhs
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcNegInf, Known.KnownFPClasses);
  EXPECT_EQ(true, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectPosOrNegInf) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %cond:_(s1) = G_LOAD %ptr(p0) :: (load (s1))
    %lhs:_(s32) = G_FCONSTANT float 0x7FF0000000000000
    %rhs:_(s32) = G_FCONSTANT float 0xFFF0000000000000
    %sel:_(s32) = G_SELECT %cond, %lhs, %rhs
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectNNaN) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %cond:_(s1) = G_LOAD %ptr(p0) :: (load (s1))
    %lhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %rhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %sel:_(s32) = nnan G_SELECT %cond, %lhs, %rhs
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(~fcNan, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectNInf) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %cond:_(s1) = G_LOAD %ptr(p0) :: (load (s1))
    %lhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %rhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %sel:_(s32) = ninf G_SELECT %cond, %lhs, %rhs
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(~fcInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectNNaNNInf) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %cond:_(s1) = G_LOAD %ptr(p0) :: (load (s1))
    %lhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %rhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %sel:_(s32) = nnan ninf G_SELECT %cond, %lhs, %rhs
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(~(fcNan | fcInf), Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFNegNInf) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fneg:_(s32) = ninf G_FNEG %val
    %copy_fneg:_(s32) = COPY %fneg
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(~fcInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFabsUnknown) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = G_FABS %val
    %copy_fabs:_(s32) = COPY %fabs
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPositive | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassVecFabsUnknown) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(<3 x s32>) = G_LOAD %ptr(p0) :: (load (<3 x s32>))
    %fabs:_(<3 x s32>) = G_FABS %val
    %copy_fabs:_(<3 x s32>) = COPY %fabs
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPositive | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFnegFabs) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = G_FABS %val
    %fneg:_(s32) = G_FNEG %fabs
    %copy_fneg:_(s32) = COPY %fneg
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcNegative | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(true, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFnegFabsNInf) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = ninf G_FABS %val
    %fneg:_(s32) = G_FNEG %fabs
    %copy_fneg:_(s32) = COPY %fneg
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ((fcNegative & ~fcNegInf) | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(true, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFnegFabsNNan) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan G_FABS %val
    %fneg:_(s32) = G_FNEG %fabs
    %copy_fneg:_(s32) = COPY %fneg
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcNegative, Known.KnownFPClasses);
  EXPECT_EQ(true, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassCopySignNNanSrc0) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %mag:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %sgn:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan G_FABS %mag
    %fcopysign:_(s32) = G_FCOPYSIGN %fabs, %sgn
    %copy_fcopysign:_(s32) = COPY %fcopysign
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(~fcNan, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassCopySignNInfSrc0_NegSign) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %mag:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %sgn:_(s32) = G_FCONSTANT float -1.0
    %fabs:_(s32) = ninf G_FLOG %mag
    %fcopysign:_(s32) = G_FCOPYSIGN %fabs, %sgn
    %copy_fcopysign:_(s32) = COPY %fcopysign
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcNegFinite | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(true, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassCopySignNInfSrc0_PosSign) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %mag:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %sgn:_(s32) = G_FCONSTANT float 1.0
    %fabs:_(s32) = ninf G_FSQRT %mag
    %fcopysign:_(s32) = G_FCOPYSIGN %fabs, %sgn
    %copy_fcopysign:_(s32) = COPY %fcopysign
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosFinite | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassUIToFP) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %cast:_(s32) = G_UITOFP %val
    %copy_cast:_(s32) = COPY %cast
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosFinite & ~fcSubnormal, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSIToFP) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %cast:_(s32) = G_SITOFP %val
    %copy_cast:_(s32) = COPY %cast
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcFinite & ~fcNegZero & ~fcSubnormal, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFAdd) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %lhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %rhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fadd:_(s32) = G_FADD %lhs, %rhs
    %copy_fadd:_(s32) = COPY %fadd
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcAllFlags, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFMul) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fmul:_(s32) = G_FMUL %val, %val
    %copy_fadd:_(s32) = COPY %fmul
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPositive | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFMulZero) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %lhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %rhs:_(s32) = G_FCONSTANT float 0.0
    %fabs:_(s32) = nnan ninf G_FABS %lhs
    %fmul:_(s32) = G_FMUL %fabs, %rhs
    %copy_fadd:_(s32) = COPY %fmul
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPositive, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFLogNeg) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan ninf G_FABS %val
    %fneg:_(s32) = nnan ninf G_FNEG %fabs
    %flog:_(s32) = G_FLOG %fneg
    %copy_flog:_(s32) = COPY %flog
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcFinite | fcNan | fcNegInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFLogPosZero) {
  StringRef MIRString = R"(
    %val:_(s32) = G_FCONSTANT float 0.0
    %flog:_(s32) = G_FLOG %val
    %copy_flog:_(s32) = COPY %flog
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcFinite | fcNegInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFLogNegZero) {
  StringRef MIRString = R"(
    %val:_(s32) = G_FCONSTANT float -0.0
    %flog:_(s32) = G_FLOG %val
    %copy_flog:_(s32) = COPY %flog
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcFinite | fcNegInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassCopy) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = G_FABS %val
    %copy:_(s32) = COPY %fabs
    %copy_copy:_(s32) = COPY %copy
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPositive | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassSelectIsFPClass) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %lhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %rhs:_(s32) = G_FCONSTANT float 0.0
    %cond:_(s1) = G_IS_FPCLASS %lhs, 96
    %sel:_(s32) = G_SELECT %cond, %lhs, %rhs 
    %copy_sel:_(s32) = COPY %sel
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcZero, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFLDExp) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %exp:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = G_FABS %val
    %fldexp:_(s32) = G_FLDEXP %fabs, %exp
    %copy_fldexp:_(s32) = COPY %fldexp
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPositive | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFPowIEvenExp) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %pwr:_(s32) = G_CONSTANT i32 2
    %fpowi:_(s32) = G_FPOWI %val, %pwr
    %copy_fpowi:_(s32) = COPY %fpowi
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPositive | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFPowIPos) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %pwr:_(s32) = G_CONSTANT i32 3
    %fabs:_(s32) = nnan ninf G_FABS %val
    %fpowi:_(s32) = G_FPOWI %fabs, %pwr
    %copy_fpowi:_(s32) = COPY %fpowi
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPositive | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFDiv) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fdiv:_(s32) = G_FDIV %val, %val
    %copy_fdiv:_(s32) = COPY %fdiv
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosNormal | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFRem) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %frem:_(s32) = G_FREM %val, %val
    %copy_frem:_(s32) = COPY %frem
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcZero | fcNan, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassShuffleVec) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %vec:_(<4 x s32>) = G_LOAD %ptr(p0) :: (load (<4 x s32>))
    %fabs:_(<4 x s32>) = nnan ninf G_FABS %vec
    %def:_(<4 x s32>) = G_IMPLICIT_DEF
    %shuf:_(<4 x s32>) = G_SHUFFLE_VECTOR %fabs(<4 x s32>), %def, shufflemask(0, 0, 0, 0)
    %copy_shuf:_(<4 x s32>) = COPY %shuf
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosFinite, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassBuildVec) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val1:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan ninf G_FABS %val1
    %val2:_(s32) = G_FCONSTANT float 3.0
    %vec:_(<2 x s32>) = G_BUILD_VECTOR %fabs, %val2
    %copy_vec:_(<2 x s32>) = COPY %vec
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosFinite, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassConcatVec) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %vec1:_(<2 x s32>) = G_LOAD %ptr(p0) :: (load (<2 x s32>))
    %c1:_(s32) = G_FCONSTANT float 1.0
    %c2:_(s32) = G_FCONSTANT float 2.0
    %vec2:_(<2 x s32>) = G_BUILD_VECTOR %c1, %c2
    %fabs1:_(<2 x s32>) = nnan ninf G_FABS %vec1
    %fabs2:_(<2 x s32>) = nnan ninf G_FABS %vec2
    %cat:_(<4 x s32>) = G_CONCAT_VECTORS %fabs1, %fabs2
    %copy_cat:_(<4 x s32>) = COPY %cat
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosFinite, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassVecExtractElem) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %vec:_(<4 x s32>) = G_LOAD %ptr(p0) :: (load (<4 x s32>))
    %fabs:_(<4 x s32>) = nnan ninf G_FABS %vec
    %idx:_(s64) = G_CONSTANT i64 1
    %extract:_(s32) = G_EXTRACT_VECTOR_ELT %fabs, %idx
    %copy_elem:_(s32) = COPY %extract
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosFinite, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassVecInsertElem) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %vec:_(<4 x s32>) = G_LOAD %ptr(p0) :: (load (<4 x s32>))
    %fabs1:_(<4 x s32>) = nnan ninf G_FABS %vec
    %elem:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs2:_(s32) = nnan ninf G_FABS %elem
    %idx:_(s64) = G_CONSTANT i64 1
    %insert:_(<4 x s32>) = G_INSERT_VECTOR_ELT %fabs1, %fabs2, %idx
    %copy_insert:_(<4 x s32>) = COPY %insert
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosFinite, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}
