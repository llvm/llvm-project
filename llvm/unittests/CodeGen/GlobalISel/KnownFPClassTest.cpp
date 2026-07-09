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

TEST_F(AArch64GISelMITest, TestFPClassCstZeroFPExt) {
  StringRef MIRString = R"(
   %c0:_(s32) = G_FCONSTANT float 0.0
   %ext:_(s64) = nnan ninf G_FPEXT %c0
   %copy_vector:_(s64) = COPY %ext
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

TEST_F(AArch64GISelMITest, TestFPClassCstVecZeroFPExt) {
  StringRef MIRString = R"(
   %c0:_(s32) = G_FCONSTANT float 0.0
   %c1:_(s32) = G_FCONSTANT float 0.0
   %c2:_(s32) = G_FCONSTANT float 0.0
   %vector:_(<3 x s32>) = G_BUILD_VECTOR %c0, %c1, %c2
   %ext:_(<3 x s64>) = nnan ninf G_FPEXT %vector
   %copy_vector:_(<3 x s64>) = COPY %ext
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

TEST_F(AArch64GISelMITest, TestFPClassCstZeroFPTrunc) {
  StringRef MIRString = R"(
   %c0:_(s64) = G_FCONSTANT double 0.0
   %trunc:_(s32) = nnan ninf G_FPTRUNC %c0
   %copy_vector:_(s32) = COPY %trunc
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosFinite | fcNegZero, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassCstVecZeroFPTrunc) {
  StringRef MIRString = R"(
   %c0:_(s64) = G_FCONSTANT double 0.0
   %c1:_(s64) = G_FCONSTANT double 0.0
   %c2:_(s64) = G_FCONSTANT double 0.0
   %vector:_(<3 x s64>) = G_BUILD_VECTOR %c0, %c1, %c2
   %trunc:_(<3 x s32>) = nnan ninf G_FPTRUNC %vector
   %copy_vector:_(<3 x s32>) = COPY %trunc
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);

  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);

  EXPECT_EQ(fcPosFinite | fcNegZero, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
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

  EXPECT_EQ(fcNegFinite | fcNan, Known.KnownFPClasses);
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

  EXPECT_EQ(fcNan | fcNegZero | fcNegNormal, Known.KnownFPClasses);
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

  EXPECT_EQ(fcNan | fcPosZero | fcPosNormal, Known.KnownFPClasses);
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

  EXPECT_EQ(fcPosZero | fcPosNormal, Known.KnownFPClasses);
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

  EXPECT_EQ(fcPosNormal | fcNegNormal | fcPosZero, Known.KnownFPClasses);
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

TEST_F(AArch64GISelMITest, TestFPClassFAdd_Zero) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %lhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %rhs:_(s32) = G_FCONSTANT float 0.0
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

  EXPECT_EQ(fcAllFlags & ~fcNegZero, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFAdd_NegZero) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %lhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %rhs:_(s32) = G_FCONSTANT float -0.0
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

TEST_F(AArch64GISelMITest, TestFPClassFstrictAdd_Zero) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %lhs:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %rhs:_(s32) = G_FCONSTANT float 0.0
    %fadd:_(s32) = G_STRICT_FADD %lhs, %rhs
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

  EXPECT_EQ(fcAllFlags & ~fcNegZero, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFMul) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %load:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %val:_(s32) = G_FREEZE %load
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

  EXPECT_EQ(fcPosZero, Known.KnownFPClasses);
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

  EXPECT_EQ(fcNan | fcNegInf | fcPosZero | fcNormal, Known.KnownFPClasses);
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

  EXPECT_EQ(fcNegInf | fcPosZero | fcNormal, Known.KnownFPClasses);
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

  EXPECT_EQ(fcNegInf | fcPosZero | fcNormal, Known.KnownFPClasses);
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

  EXPECT_EQ(~fcNegative, Known.KnownFPClasses);
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

  EXPECT_EQ(fcPositive, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFPowIInf) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %load:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %val:_(s32) = G_FREEZE %load
    %x:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %finite:_(s32) = ninf G_FNEG %val
    %normal:_(s32) = G_FCONSTANT float 2.0
    %zero_or_nan:_(s32) = G_FREM %val, %val
    %nonneg_mask:_(s32) = G_CONSTANT i32 2147483647
    %nonneg:_(s32) = G_AND %x, %nonneg_mask
    %one:_(s32) = G_CONSTANT i32 1
    %two:_(s32) = G_CONSTANT i32 2
    %negone:_(s32) = G_CONSTANT i32 -1
    %fpowi0:_(s32) = G_FPOWI %finite, %one
    %copy_fpowi0:_(s32) = COPY %fpowi0
    %fpowi1:_(s32) = G_FPOWI %finite, %two
    %copy_fpowi1:_(s32) = COPY %fpowi1
    %fpowi2:_(s32) = G_FPOWI %normal, %negone
    %copy_fpowi2:_(s32) = COPY %fpowi2
    %fpowi3:_(s32) = G_FPOWI %zero_or_nan, %nonneg
    %copy_fpowi3:_(s32) = COPY %fpowi3
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  GISelValueTracking Info(*MF);

  // powi(finite, 1)  -->  ~fcInf
  Register CopyReg0 = Copies[Copies.size() - 4];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  KnownFPClass Known0 = Info.computeKnownFPClass(SrcReg0, fcAllFlags);
  KnownFPClass KnownInf0 = Info.computeKnownFPClass(SrcReg0, fcInf);
  EXPECT_EQ(~fcInf, Known0.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known0.SignBit);
  EXPECT_TRUE(KnownInf0.isKnownNeverInfinity());

  // powi(finite, 2)  -->  fcPositive | fcNan
  Register CopyReg1 = Copies[Copies.size() - 3];
  MachineInstr *FinalCopy1 = MRI->getVRegDef(CopyReg1);
  Register SrcReg1 = FinalCopy1->getOperand(1).getReg();
  KnownFPClass Known1 = Info.computeKnownFPClass(SrcReg1, fcAllFlags);
  KnownFPClass KnownInf1 = Info.computeKnownFPClass(SrcReg1, fcInf);
  EXPECT_EQ(fcPositive | fcNan, Known1.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known1.SignBit);
  EXPECT_FALSE(KnownInf1.isKnownNeverInfinity());

  // powi(normal, -1)  -->  fcPosFinite
  Register CopyReg2 = Copies[Copies.size() - 2];
  MachineInstr *FinalCopy2 = MRI->getVRegDef(CopyReg2);
  Register SrcReg2 = FinalCopy2->getOperand(1).getReg();
  KnownFPClass Known2 = Info.computeKnownFPClass(SrcReg2, fcAllFlags);
  KnownFPClass KnownInf2 = Info.computeKnownFPClass(SrcReg2, fcInf);
  EXPECT_EQ(fcPosFinite, Known2.KnownFPClasses);
  EXPECT_EQ(false, Known2.SignBit);
  EXPECT_TRUE(KnownInf2.isKnownNeverInfinity());

  // powi(zero_or_nan, nonneg)  -->  ~fcInf
  Register CopyReg3 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy3 = MRI->getVRegDef(CopyReg3);
  Register SrcReg3 = FinalCopy3->getOperand(1).getReg();
  KnownFPClass KnownInf3 = Info.computeKnownFPClass(SrcReg3, fcInf);
  EXPECT_TRUE(KnownInf3.isKnownNeverInfinity());

  // TODO: Add powi(0/nan, exp), exp > 0  -->  fcNan | fcZero | fcPosNormal
}

TEST_F(AArch64GISelMITest, TestFPClassFDiv) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %load:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %val:_(s32) = G_FREEZE %load
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

TEST_F(AArch64GISelMITest, TestFPClassFDiv_Inf) {
  StringRef MIRString = R"(
    %lhs:_(s32) = G_FCONSTANT float 1.0
    %rhs:_(s32) = G_FCONSTANT float 0.0
    %fdiv:_(s32) = G_FDIV %lhs, %rhs
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

  EXPECT_EQ(fcPosInf, Known.KnownFPClasses);
  EXPECT_EQ(false, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFRem) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %load:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %val:_(s32) = G_FREEZE %load
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

TEST_F(AArch64GISelMITest, TestFPClassFRemSelf_KnownFiniteNonZero) {
  // X % X where X is a known-finite, known-nonzero value should produce
  // exactly [+-]0.0 (no NaN possible).
  StringRef MIRString = R"(
    %val:_(s32) = G_FCONSTANT float 2.0
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

  // 2.0 % 2.0 = 0.0 exactly — NaN is impossible since 2.0 is finite and
  // nonzero.
  EXPECT_EQ(fcZero, Known.KnownFPClasses);
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

TEST_F(AArch64GISelMITest, TestFPClassFSinh) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fsinh:_(s32) = G_FSINH %val
    %copy:_(s32) = COPY %fsinh
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

TEST_F(AArch64GISelMITest, TestFPClassFSinhPos) {
  // sinh is sign-preserving: non-negative input → non-negative output.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan ninf G_FABS %val
    %fsinh:_(s32) = G_FSINH %fabs
    %copy:_(s32) = COPY %fsinh
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

TEST_F(AArch64GISelMITest, TestFPClassFCosh) {
  // cosh(x) >= 1 for all real x; never negative.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fcosh:_(s32) = G_FCOSH %val
    %copy:_(s32) = COPY %fcosh
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

TEST_F(AArch64GISelMITest, TestFPClassFCoshNNaN) {
  // cosh of a non-NaN source is non-NaN (and non-negative).
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan G_FABS %val
    %fcosh:_(s32) = G_FCOSH %fabs
    %copy:_(s32) = COPY %fcosh
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

TEST_F(AArch64GISelMITest, TestFPClassFTanh) {
  // tanh is bounded to (-1, 1): never Inf.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %ftanh:_(s32) = G_FTANH %val
    %copy:_(s32) = COPY %ftanh
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);
  EXPECT_EQ(fcAllFlags & ~fcInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFTanhPos) {
  // tanh is sign-preserving and bounded to (-1,1): non-negative finite output.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan ninf G_FABS %val
    %ftanh:_(s32) = G_FTANH %fabs
    %copy:_(s32) = COPY %ftanh
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

TEST_F(AArch64GISelMITest, TestFPClassFAsin) {
  // asin is bounded to [-π/2, π/2]: never Inf.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fasin:_(s32) = G_FASIN %val
    %copy:_(s32) = COPY %fasin
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);
  EXPECT_EQ(fcAllFlags & ~fcInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFAsinPos) {
  // asin is sign-preserving and bounded: non-negative finite output.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan ninf G_FABS %val
    %fasin:_(s32) = G_FASIN %fabs
    %copy:_(s32) = COPY %fasin
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

TEST_F(AArch64GISelMITest, TestFPClassFAcos) {
  // acos is bounded to [0, π]: never Inf, never negative.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %facos:_(s32) = G_FACOS %val
    %copy:_(s32) = COPY %facos
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
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFAtan) {
  // atan is bounded to (-π/2, π/2): never Inf (atan(±Inf) = ±π/2, finite).
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fatan:_(s32) = G_FATAN %val
    %copy:_(s32) = COPY %fatan
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);
  EXPECT_EQ(fcAllFlags & ~fcInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFAtanPos) {
  // atan is sign-preserving and bounded: non-negative finite output.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan ninf G_FABS %val
    %fatan:_(s32) = G_FATAN %fabs
    %copy:_(s32) = COPY %fatan
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

TEST_F(AArch64GISelMITest, TestFPClassFTan) {
  // tan(±Inf) = NaN, tan(finite) = finite: never Inf.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %ftan:_(s32) = G_FTAN %val
    %copy:_(s32) = COPY %ftan
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);
  EXPECT_EQ(fcAllFlags & ~fcInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFTanNNaN) {
  // tan of a non-NaN, non-Inf source is non-NaN and non-Inf.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = nnan ninf G_FABS %val
    %ftan:_(s32) = G_FTAN %fabs
    %copy:_(s32) = COPY %ftan
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);
  EXPECT_EQ(fcFinite, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFAtan2) {
  // atan2 result is in (-π, π]: never Inf.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %y:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %x:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fatan2:_(s32) = G_FATAN2 %y, %x
    %copy:_(s32) = COPY %fatan2
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);
  EXPECT_EQ(fcAllFlags & ~fcInf, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

TEST_F(AArch64GISelMITest, TestFPClassFAtan2NNaN) {
  // atan2 with two non-NaN inputs is non-NaN and non-Inf.
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %y:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %x:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs_y:_(s32) = nnan ninf G_FABS %y
    %fabs_x:_(s32) = nnan ninf G_FABS %x
    %fatan2:_(s32) = G_FATAN2 %fabs_y, %fabs_x
    %copy:_(s32) = COPY %fatan2
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);
  EXPECT_EQ(fcFinite, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}

// isAbsoluteValueULEOne: x - floor(x) is in [0, 1), so multiplying a known-
// finite value by it cannot overflow to infinity.
TEST_F(AArch64GISelMITest, TestFPClassFMulAbsULEOne) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %x:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %floor:_(s32) = G_FFLOOR %x
    %fract:_(s32) = G_FSUB %x, %floor
    %finite:_(s32) = nnan ninf G_FABS %val
    %fmul:_(s32) = G_FMUL %finite, %fract
    %copy:_(s32) = COPY %fmul
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

// G_FMA with A == B (and A guaranteed not-undef): the multiply part is a
// square, so the result is known non-negative (never fcNegative).
TEST_F(AArch64GISelMITest, TestFPClassFMASelfSquare) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %load:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %val:_(s32) = G_FREEZE %load
    %c:_(s32) = G_FCONSTANT float 1.0
    %fma:_(s32) = G_FMA %val, %val, %c
    %copy:_(s32) = COPY %fma
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelValueTracking Info(*MF);
  KnownFPClass Known = Info.computeKnownFPClass(SrcReg);
  EXPECT_EQ(fcNan | fcPosInf | fcPosNormal, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.SignBit);
}
