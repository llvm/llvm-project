//===- KnownBitsTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

TEST_F(AArch64GISelMITest, TestKnownBitsBuildVector) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load (s8))
   %mask0:_(s8) = G_CONSTANT i8 24
   %mask1:_(s8) = G_CONSTANT i8 224
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %mask2:_(s8) = G_CONSTANT i8 146
   %mask3:_(s8) = G_CONSTANT i8 36
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %vector:_(<2 x s8>) = G_BUILD_VECTOR %val0, %val1
   %copy_vector:_(<2 x s8>) = COPY %vector
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  // BuildVector KnownBits takes common bits of all elements.
  //        111??000
  // common ?01?01?0
  //      = ??1????0
  EXPECT_EQ(0x20u, Res.One.getZExtValue());
  EXPECT_EQ(0x01u, Res.Zero.getZExtValue());
}

// Vector KnownBits track bits that are common for all vector scalar elements.
// For tests below KnownBits analysis is same as for scalar/pointer types, tests
// are mostly copied from KnownBitsTest.cpp using splat vectors and have the
// same result.

TEST_F(AArch64GISelMITest, TestKnownBitsVectorCstPHI) {
  StringRef MIRString = R"(
   bb.10:
   %10:_(s8) = G_CONSTANT i8 3
   %11:_(<2 x s8>) = G_BUILD_VECTOR %10:_(s8), %10:_(s8)
   %12:_(s1) = G_IMPLICIT_DEF
   G_BRCOND %12(s1), %bb.11
   G_BR %bb.12

   bb.11:
   %13:_(s8) = G_CONSTANT i8 2
   %14:_(<2 x s8>) = G_BUILD_VECTOR %13:_(s8), %13:_(s8)
   G_BR %bb.12

   bb.12:
   %15:_(<2 x s8>) = PHI %11(<2 x s8>), %bb.10, %14(<2 x s8>), %bb.11
   %16:_(<2 x s8>) = COPY %15
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  Register DstReg = FinalCopy->getOperand(0).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)2, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0xfc, Res.Zero.getZExtValue());

  KnownBits Res2 = Info.getKnownBits(DstReg);
  EXPECT_EQ(Res.One.getZExtValue(), Res2.One.getZExtValue());
  EXPECT_EQ(Res.Zero.getZExtValue(), Res2.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorCstPHIToNonGenericReg) {
  StringRef MIRString = R"(
   bb.10:
   %10:gpr32 = MOVi32imm 771
   %11:_(s1) = G_IMPLICIT_DEF
   G_BRCOND %11(s1), %bb.11
   G_BR %bb.12

   bb.11:
   %12:_(s16) = G_CONSTANT i16 2
   %13:_(<2 x s16>) = G_BUILD_VECTOR %12:_(s16), %12:_(s16)
   G_BR %bb.12

   bb.12:
   %15:_(<2 x s16>) = PHI %10, %bb.10, %13(<2 x s16>), %bb.11
   %16:_(<2 x s16>) = COPY %15
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  Register DstReg = FinalCopy->getOperand(0).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)0, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0, Res.Zero.getZExtValue());

  KnownBits Res2 = Info.getKnownBits(DstReg);
  EXPECT_EQ(Res.One.getZExtValue(), Res2.One.getZExtValue());
  EXPECT_EQ(Res.Zero.getZExtValue(), Res2.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorUnknownPHI) {
  StringRef MIRString = R"(
   bb.10:
   %10:_(<2 x s32>) = G_BITCAST %0
   %11:_(s1) = G_IMPLICIT_DEF
   G_BRCOND %11(s1), %bb.11
   G_BR %bb.12

   bb.11:
   %12:_(s32) = G_CONSTANT i32 2
   %13:_(<2 x s32>) = G_BUILD_VECTOR %12:_(s32), %12:_(s32)
   G_BR %bb.12

   bb.12:
   %14:_(<2 x s32>) = PHI %10(<2 x s32>), %bb.10, %13(<2 x s32>), %bb.11
   %15:_(<2 x s32>) = COPY %14
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  Register DstReg = FinalCopy->getOperand(0).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)0, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0, Res.Zero.getZExtValue());

  KnownBits Res2 = Info.getKnownBits(DstReg);
  EXPECT_EQ(Res.One.getZExtValue(), Res2.One.getZExtValue());
  EXPECT_EQ(Res.Zero.getZExtValue(), Res2.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorCstPHIWithLoop) {
  StringRef MIRString = R"(
   bb.10:
   %10:_(s8) = G_CONSTANT i8 3
   %11:_(<2 x s8>) = G_BUILD_VECTOR %10:_(s8), %10:_(s8)
   %12:_(s1) = G_IMPLICIT_DEF
   G_BRCOND %12(s1), %bb.11
   G_BR %bb.12

   bb.11:
   %13:_(s8) = G_CONSTANT i8 2
   %14:_(<2 x s8>) = G_BUILD_VECTOR %13:_(s8), %13:_(s8)
   G_BR %bb.12

   bb.12:
   %15:_(<2 x s8>) = PHI %11(<2 x s8>), %bb.10, %14(<2 x s8>), %bb.11, %16(<2 x s8>), %bb.12
   %16:_(<2 x s8>) = COPY %15
   G_BR %bb.12
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  Register DstReg = FinalCopy->getOperand(0).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)0, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0, Res.Zero.getZExtValue());

  KnownBits Res2 = Info.getKnownBits(DstReg);
  EXPECT_EQ(Res.One.getZExtValue(), Res2.One.getZExtValue());
  EXPECT_EQ(Res.Zero.getZExtValue(), Res2.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorDecreasingCstPHIWithLoop) {
  StringRef MIRString = R"(
   bb.10:
   %10:_(s8) = G_CONSTANT i8 5
   %11:_(<2 x s8>) = G_BUILD_VECTOR %10:_(s8), %10:_(s8)
   %12:_(s8) = G_CONSTANT i8 1
   %16:_(<2 x s8>) = G_BUILD_VECTOR %12:_(s8), %12:_(s8)

   bb.12:
   %13:_(<2 x s8>) = PHI %11(<2 x s8>), %bb.10, %14(<2 x s8>), %bb.12
   %14:_(<2 x s8>) = G_LSHR %13, %16
   %15:_(<2 x s8>) = COPY %14
   G_BR %bb.12
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  Register DstReg = FinalCopy->getOperand(0).getReg();
  GISelKnownBits Info(*MF, /*MaxDepth=*/24);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)0, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0xC0, Res.Zero.getZExtValue());

  KnownBits Res2 = Info.getKnownBits(DstReg);
  EXPECT_EQ(Res.One.getZExtValue(), Res2.One.getZExtValue());
  EXPECT_EQ(Res.Zero.getZExtValue(), Res2.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorAND) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 52
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 10
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s8) = G_CONSTANT i8 32
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 24
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %and:_(<2 x s8>) = G_AND %val0, %val1
   %copy_and:_(<2 x s8>) = COPY %and
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(0x08u, Res.One.getZExtValue());
  EXPECT_EQ(0xC7u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorOR) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 52
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 10
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s8) = G_CONSTANT i8 32
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 24
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %or:_(<2 x s8>) = G_OR %val0, %val1
   %copy_or:_(<2 x s8>) = COPY %or
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(0x1Au, Res.One.getZExtValue());
  EXPECT_EQ(0xC1u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorXOR) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 52
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 10
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s8) = G_CONSTANT i8 32
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 24
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %xor:_(<2 x s8>) = G_XOR %val0, %val1
   %copy_xor:_(<2 x s8>) = COPY %xor
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(0x02u, Res.One.getZExtValue());
  EXPECT_EQ(0xC9u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorXORConstant) {
  StringRef MIRString = R"(
   %3:_(s8) = G_CONSTANT i8 4
   %4:_(<2 x s8>) = G_BUILD_VECTOR %3:_(s8), %3:_(s8)
   %5:_(s8) = G_CONSTANT i8 7
   %6:_(<2 x s8>) = G_BUILD_VECTOR %5:_(s8), %5:_(s8)
   %7:_(<2 x s8>) = G_XOR %4, %6
   %8:_(<2 x s8>) = COPY %7

   %9:_(s8) = G_CONSTANT i8 12
   %10:_(<2 x s8>) = G_BUILD_VECTOR %3:_(s8), %9:_(s8)
   %11:_(<2 x s8>) = G_XOR %10, %6
   %12:_(<2 x s8>) = COPY %11
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  GISelKnownBits Info(*MF);
  Register CopySplatReg = Copies[Copies.size() - 2];
  MachineInstr *FinalSplatCopy = MRI->getVRegDef(CopySplatReg);
  Register SrcSplatReg = FinalSplatCopy->getOperand(1).getReg();
  KnownBits ResNonSplat = Info.getKnownBits(SrcSplatReg);
  EXPECT_EQ(3u, ResNonSplat.One.getZExtValue());
  EXPECT_EQ(252u, ResNonSplat.Zero.getZExtValue());

  Register CopyNonSplatReg = Copies[Copies.size() - 1];
  MachineInstr *FinalNonSplatCopy = MRI->getVRegDef(CopyNonSplatReg);
  Register SrcNonSplatReg = FinalNonSplatCopy->getOperand(1).getReg();
  KnownBits ResSplat = Info.getKnownBits(SrcNonSplatReg);
  EXPECT_EQ(3u, ResSplat.One.getZExtValue());
  EXPECT_EQ(244u, ResSplat.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorASHR) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 38
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 202
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %cst0:_(s8) = G_CONSTANT i8 2
   %cst0_splat:_(<2 x s8>) = G_BUILD_VECTOR %cst0, %cst0
   %ashr0:_(<2 x s8>) = G_ASHR %val0, %cst0_splat
   %copy_ashr0:_(<2 x s8>) = COPY %ashr0

   %mask2:_(s8) = G_CONSTANT i8 204
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 18
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %ashr1:_(<2 x s8>) = G_ASHR %val1, %cst0_splat
   %copy_ashr1:_(<2 x s8>) = COPY %ashr1
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg0 = Copies[Copies.size() - 2];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  EXPECT_EQ(0xF2u, Res0.One.getZExtValue());
  EXPECT_EQ(0x04u, Res0.Zero.getZExtValue());

  Register CopyReg1 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy1 = MRI->getVRegDef(CopyReg1);
  Register SrcReg1 = FinalCopy1->getOperand(1).getReg();
  KnownBits Res1 = Info.getKnownBits(SrcReg1);
  EXPECT_EQ(0x04u, Res1.One.getZExtValue());
  EXPECT_EQ(0x08u, Res1.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorLSHR) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 38
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 202
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %cst0:_(s8) = G_CONSTANT i8 2
   %cst0_splat:_(<2 x s8>) = G_BUILD_VECTOR %cst0, %cst0
   %lshr0:_(<2 x s8>) = G_LSHR %val0, %cst0_splat
   %copy_lshr0:_(<2 x s8>) = COPY %lshr0

   %mask2:_(s8) = G_CONSTANT i8 204
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 18
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %lshr1:_(<2 x s8>) = G_LSHR %val1, %cst0_splat
   %copy_lshr1:_(<2 x s8>) = COPY %lshr1
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg0 = Copies[Copies.size() - 2];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  EXPECT_EQ(0x32u, Res0.One.getZExtValue());
  EXPECT_EQ(0xC4u, Res0.Zero.getZExtValue());

  Register CopyReg1 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy1 = MRI->getVRegDef(CopyReg1);
  Register SrcReg1 = FinalCopy1->getOperand(1).getReg();
  KnownBits Res1 = Info.getKnownBits(SrcReg1);
  EXPECT_EQ(0x04u, Res1.One.getZExtValue());
  EXPECT_EQ(0xC8u, Res1.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorSHL) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 51
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 72
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val:_(<2 x s8>) = G_OR %tmp, %mask1_splat
   %cst:_(s8) = G_CONSTANT i8 3
   %cst_splat:_(<2 x s8>) = G_BUILD_VECTOR %cst, %cst
   %shl:_(<2 x s8>) = G_SHL %val, %cst_splat
   %copy_shl:_(<2 x s8>) = COPY %shl
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(0x40u, Res.One.getZExtValue());
  EXPECT_EQ(0x27u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorADD) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s16>) = G_LOAD %ptr(p0) :: (load (<2 x s16>))
   %mask0:_(s16) = G_CONSTANT i16 4642
   %mask0_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s16) = G_CONSTANT i16 9536
   %mask1_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s16>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s16>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s16) = G_CONSTANT i16 4096
   %mask2_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s16) = G_CONSTANT i16 371
   %mask3_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s16>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s16>) = G_OR %tmp1, %mask3_splat
   %add:_(<2 x s16>) = G_ADD %val0, %val1
   %copy_add:_(<2 x s16>) = COPY %add
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(0x0091u, Res.One.getZExtValue());
  EXPECT_EQ(0x8108u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorSUB) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s16>) = G_LOAD %ptr(p0) :: (load (<2 x s16>))
   %mask0:_(s16) = G_CONSTANT i16 4642
   %mask0_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s16) = G_CONSTANT i16 9536
   %mask1_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s16>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s16>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s16) = G_CONSTANT i16 4096
   %mask2_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s16) = G_CONSTANT i16 371
   %mask3_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s16>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s16>) = G_OR %tmp1, %mask3_splat
   %sub:_(<2 x s16>) = G_SUB %val0, %val1
   %copy_sub:_(<2 x s16>) = COPY %sub
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(0x01CDu, Res.One.getZExtValue());
  EXPECT_EQ(0xC810u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorMUL) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s16>) = G_LOAD %ptr(p0) :: (load (<2 x s16>))
   %mask0:_(s16) = G_CONSTANT i16 4
   %mask0_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s16) = G_CONSTANT i16 18
   %mask1_splat:_(<2 x s16>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp:_(<2 x s16>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s16>) = G_OR %tmp, %mask1_splat
   %cst:_(s16) = G_CONSTANT i16 12
   %cst_splat:_(<2 x s16>) = G_BUILD_VECTOR %cst, %cst
   %mul:_(<2 x s16>) = G_MUL %val0, %cst_splat
   %copy_mul:_(<2 x s16>) = COPY %mul
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(0x0008u, Res.One.getZExtValue());
  EXPECT_EQ(0xFE07u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorSelect) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 24
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 224
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s8) = G_CONSTANT i8 146
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 36
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %cond:_(s1) = G_CONSTANT i1 false
   %cond_splat:_(<2 x s1>) = G_BUILD_VECTOR %cond, %cond
   %select:_(<2 x s8>) = G_SELECT %cond_splat, %val0, %val1
   %copy_select:_(<2 x s8>) = COPY %select
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(0x20u, Res.One.getZExtValue());
  EXPECT_EQ(0x01u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestVectorSignBitIsZero) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  const LLT V2S32 = LLT::fixed_vector(2, 32);
  // Vector buildConstant makes splat G_BUILD_VECTOR instruction.
  auto SignBit = B.buildConstant(V2S32, 0x80000000);
  auto Zero = B.buildConstant(V2S32, 0);

  const LLT S32 = LLT::scalar(32);
  auto NonSplat =
      B.buildBuildVector(V2S32, {B.buildConstant(S32, 1).getReg(0),
                                 B.buildConstant(S32, 2).getReg(0)});
  auto NonSplat2 =
      B.buildBuildVector(V2S32, {B.buildConstant(S32, 0x80000000).getReg(0),
                                 B.buildConstant(S32, 0x80000004).getReg(0)});
  // signBitIsZero is true for elt 0 and false for elt 1 GISelKnownBits takes
  // common bits so this is false.
  auto NonSplat3 =
      B.buildBuildVector(V2S32, {B.buildConstant(S32, 0x80000000).getReg(0),
                                 B.buildConstant(S32, 0x8).getReg(0)});
  GISelKnownBits KnownBits(*MF);

  EXPECT_TRUE(KnownBits.signBitIsZero(Zero.getReg(0)));
  EXPECT_FALSE(KnownBits.signBitIsZero(SignBit.getReg(0)));
  EXPECT_TRUE(KnownBits.signBitIsZero(NonSplat.getReg(0)));
  EXPECT_FALSE(KnownBits.signBitIsZero(NonSplat2.getReg(0)));
  EXPECT_FALSE(KnownBits.signBitIsZero(NonSplat3.getReg(0)));
}

TEST_F(AArch64GISelMITest, TestVectorNumSignBitsConstant) {
  StringRef MIRString = R"(
   %3:_(s8) = G_CONSTANT i8 1
   %4:_(<2 x s8>) = G_BUILD_VECTOR %3:_(s8), %3:_(s8)
   %5:_(<2 x s8>) = COPY %4

   %6:_(s8) = G_CONSTANT i8 -1
   %7:_(<2 x s8>) = G_BUILD_VECTOR %6:_(s8), %6:_(s8)
   %8:_(<2 x s8>) = COPY %7

   %9:_(s8) = G_CONSTANT i8 127
   %10:_(<2 x s8>) = G_BUILD_VECTOR %9:_(s8), %9:_(s8)
   %11:_(<2 x s8>) = COPY %10

   %12:_(s8) = G_CONSTANT i8 32
   %13:_(<2 x s8>) = G_BUILD_VECTOR %12:_(s8), %12:_(s8)
   %14:_(<2 x s8>) = COPY %13

   %15:_(s8) = G_CONSTANT i8 -32
   %16:_(<2 x s8>) = G_BUILD_VECTOR %15:_(s8), %15:_(s8)
   %17:_(<2 x s8>) = COPY %16

   %18:_(<2 x s8>) = G_BUILD_VECTOR %6:_(s8), %15:_(s8)
   %19:_(<2 x s8>) = COPY %18

   %20:_(<2 x s8>) = G_BUILD_VECTOR %12:_(s8), %15:_(s8)
   %21:_(<2 x s8>) = COPY %20
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg1 = Copies[Copies.size() - 7];
  Register CopyRegNeg1 = Copies[Copies.size() - 6];
  Register CopyReg127 = Copies[Copies.size() - 5];
  Register CopyReg32 = Copies[Copies.size() - 4];
  Register CopyRegNeg32 = Copies[Copies.size() - 3];
  Register NonSplatSameSign = Copies[Copies.size() - 2];
  Register NonSplatDifferentSign = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  // If it is known that all elts have same sign looks at common bits and
  // effectively returns smallest NumSignBits of all the elts. Otherwise returns
  // default value 1.
  EXPECT_EQ(7u, Info.computeNumSignBits(CopyReg1));
  EXPECT_EQ(8u, Info.computeNumSignBits(CopyRegNeg1));
  EXPECT_EQ(1u, Info.computeNumSignBits(CopyReg127));
  EXPECT_EQ(2u, Info.computeNumSignBits(CopyReg32));
  EXPECT_EQ(3u, Info.computeNumSignBits(CopyRegNeg32));
  EXPECT_EQ(3u, Info.computeNumSignBits(NonSplatSameSign));
  EXPECT_EQ(1u, Info.computeNumSignBits(NonSplatDifferentSign));
}

TEST_F(AArch64GISelMITest, TestVectorNumSignBitsSext) {
  StringRef MIRString = R"(
   %3:_(p0) = G_IMPLICIT_DEF
   %4:_(<2 x s8>) = G_LOAD %3 :: (load (<2 x s8>))
   %5:_(<2 x s32>) = G_SEXT %4
   %6:_(<2 x s32>) = COPY %5

   %7:_(s8) = G_CONSTANT i8 -1
   %8:_(<2 x s8>) = G_BUILD_VECTOR %7:_(s8), %7:_(s8)
   %9:_(<2 x s32>) = G_SEXT %8
   %10:_(<2 x s32>) = COPY %9

   %11:_(s8) = G_CONSTANT i8 -10
   %12:_(<2 x s8>) = G_BUILD_VECTOR %7:_(s8), %11:_(s8)
   %13:_(<2 x s32>) = G_SEXT %12
   %14:_(<2 x s32>) = COPY %13
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopySextLoad = Copies[Copies.size() - 3];
  Register CopySextNeg1 = Copies[Copies.size() - 2];
  Register CopySextNonSplat = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  EXPECT_EQ(25u, Info.computeNumSignBits(CopySextLoad));
  EXPECT_EQ(32u, Info.computeNumSignBits(CopySextNeg1));
  EXPECT_EQ(28u, Info.computeNumSignBits(CopySextNonSplat));
}

TEST_F(AArch64GISelMITest, TestVectorNumSignBitsSextInReg) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %load2x4:_(<2 x s32>) = G_LOAD %ptr :: (load (<2 x s32>))

   %inreg7:_(<2 x s32>) = G_SEXT_INREG %load2x4, 7
   %copy_inreg7:_(<2 x s32>) = COPY %inreg7

   %inreg8:_(<2 x s32>) = G_SEXT_INREG %load2x4, 8
   %copy_inreg8:_(<2 x s32>) = COPY %inreg8

   %inreg9:_(<2 x s32>) = G_SEXT_INREG %load2x4, 9
   %copy_inreg9:_(<2 x s32>) = COPY %inreg9

   %inreg31:_(<2 x s32>) = G_SEXT_INREG %load2x4, 31
   %copy_inreg31:_(<2 x s32>) = COPY %inreg31

   %load2x1:_(<2 x s8>) = G_LOAD %ptr :: (load (<2 x s8>))
   %sext_load2x1:_(<2 x s32>) = G_SEXT %load2x1

   %inreg6_sext:_(<2 x s32>) = G_SEXT_INREG %sext_load2x1, 6
   %copy_inreg6_sext:_(<2 x s32>) = COPY %inreg6_sext

   %inreg7_sext:_(<2 x s32>) = G_SEXT_INREG %sext_load2x1, 7
   %copy_inreg7_sext:_(<2 x s32>) = COPY %inreg7_sext

   %inreg8_sext:_(<2 x s32>) = G_SEXT_INREG %sext_load2x1, 8
   %copy_inreg8_sext:_(<2 x s32>) = COPY %inreg8_sext

   %inreg9_sext:_(<2 x s32>) = G_SEXT_INREG %sext_load2x1, 9
   %copy_inreg9_sext:_(<2 x s32>) = COPY %inreg9_sext

   %inreg31_sext:_(<2 x s32>) = G_SEXT_INREG %sext_load2x1, 31
   %copy_inreg31_sext:_(<2 x s32>) = COPY %inreg31_sext
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyInReg7 = Copies[Copies.size() - 9];
  Register CopyInReg8 = Copies[Copies.size() - 8];
  Register CopyInReg9 = Copies[Copies.size() - 7];
  Register CopyInReg31 = Copies[Copies.size() - 6];

  Register CopyInReg6Sext = Copies[Copies.size() - 5];
  Register CopyInReg7Sext = Copies[Copies.size() - 4];
  Register CopyInReg8Sext = Copies[Copies.size() - 3];
  Register CopyInReg9Sext = Copies[Copies.size() - 2];
  Register CopyInReg31Sext = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  EXPECT_EQ(26u, Info.computeNumSignBits(CopyInReg7));
  EXPECT_EQ(25u, Info.computeNumSignBits(CopyInReg8));
  EXPECT_EQ(24u, Info.computeNumSignBits(CopyInReg9));
  EXPECT_EQ(2u, Info.computeNumSignBits(CopyInReg31));

  EXPECT_EQ(27u, Info.computeNumSignBits(CopyInReg6Sext));
  EXPECT_EQ(26u, Info.computeNumSignBits(CopyInReg7Sext));
  EXPECT_EQ(25u, Info.computeNumSignBits(CopyInReg8Sext));
  EXPECT_EQ(25u, Info.computeNumSignBits(CopyInReg9Sext));
  EXPECT_EQ(25u, Info.computeNumSignBits(CopyInReg31Sext));
}

TEST_F(AArch64GISelMITest, TestNumSignBitsVectorAssertSext) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %load2x4:_(<2 x s32>) = G_LOAD %ptr :: (load (<2 x s32>))

   %assert_sext1:_(<2 x s32>) = G_ASSERT_SEXT %load2x4, 1
   %copy_assert_sext1:_(<2 x s32>) = COPY %assert_sext1

   %assert_sext7:_(<2 x s32>) = G_ASSERT_SEXT %load2x4, 7
   %copy_assert_sext7:_(<2 x s32>) = COPY %assert_sext7

   %assert_sext8:_(<2 x s32>) = G_ASSERT_SEXT %load2x4, 8
   %copy_assert_sext8:_(<2 x s32>) = COPY %assert_sext8

   %assert_sext9:_(<2 x s32>) = G_ASSERT_SEXT %load2x4, 9
   %copy_assert_sext9:_(<2 x s32>) = COPY %assert_sext9

   %assert_sext31:_(<2 x s32>) = G_ASSERT_SEXT %load2x4, 31
   %copy_assert_sext31:_(<2 x s32>) = COPY %assert_sext31

   %load2x1:_(<2 x s8>) = G_LOAD %ptr :: (load (<2 x s8>))
   %sext_load2x1:_(<2 x s32>) = G_SEXT %load2x1

   %assert_sext6_sext:_(<2 x s32>) = G_ASSERT_SEXT %sext_load2x1, 6
   %copy_assert_sext6_sext:_(<2 x s32>) = COPY %assert_sext6_sext

   %assert_sext7_sext:_(<2 x s32>) = G_ASSERT_SEXT %sext_load2x1, 7
   %copy_assert_sext7_sext:_(<2 x s32>) = COPY %assert_sext7_sext

   %assert_sext8_sext:_(<2 x s32>) = G_ASSERT_SEXT %sext_load2x1, 8
   %copy_assert_sext8_sext:_(<2 x s32>) = COPY %assert_sext8_sext

   %assert_sext9_sext:_(<2 x s32>) = G_ASSERT_SEXT %sext_load2x1, 9
   %copy_assert_sext9_sext:_(<2 x s32>) = COPY %assert_sext9_sext

   %assert_sext31_sext:_(<2 x s32>) = G_ASSERT_SEXT %sext_load2x1, 31
   %copy_assert_sext31_sext:_(<2 x s32>) = COPY %assert_sext31_sext
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyInReg1 = Copies[Copies.size() - 10];
  Register CopyInReg7 = Copies[Copies.size() - 9];
  Register CopyInReg8 = Copies[Copies.size() - 8];
  Register CopyInReg9 = Copies[Copies.size() - 7];
  Register CopyInReg31 = Copies[Copies.size() - 6];

  Register CopyInReg6Sext = Copies[Copies.size() - 5];
  Register CopyInReg7Sext = Copies[Copies.size() - 4];
  Register CopyInReg8Sext = Copies[Copies.size() - 3];
  Register CopyInReg9Sext = Copies[Copies.size() - 2];
  Register CopyInReg31Sext = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  EXPECT_EQ(32u, Info.computeNumSignBits(CopyInReg1));
  EXPECT_EQ(26u, Info.computeNumSignBits(CopyInReg7));
  EXPECT_EQ(25u, Info.computeNumSignBits(CopyInReg8));
  EXPECT_EQ(24u, Info.computeNumSignBits(CopyInReg9));
  EXPECT_EQ(2u, Info.computeNumSignBits(CopyInReg31));

  EXPECT_EQ(27u, Info.computeNumSignBits(CopyInReg6Sext));
  EXPECT_EQ(26u, Info.computeNumSignBits(CopyInReg7Sext));
  EXPECT_EQ(25u, Info.computeNumSignBits(CopyInReg8Sext));
  EXPECT_EQ(25u, Info.computeNumSignBits(CopyInReg9Sext));
  EXPECT_EQ(25u, Info.computeNumSignBits(CopyInReg31Sext));
}

TEST_F(AArch64GISelMITest, TestVectorNumSignBitsTrunc) {
  StringRef MIRString = R"(
   %3:_(p0) = G_IMPLICIT_DEF
   %4:_(<2 x s32>) = G_LOAD %3 :: (load (<2 x s32>))
   %5:_(<2 x s8>) = G_TRUNC %4
   %6:_(<2 x s8>) = COPY %5

   %7:_(s32) = G_CONSTANT i32 -1
   %8:_(<2 x s32>) = G_BUILD_VECTOR %7:_(s32), %7:_(s32)
   %9:_(<2 x s8>) = G_TRUNC %8
   %10:_(<2 x s8>) = COPY %9

   %11:_(s32) = G_CONSTANT i32 7
   %12:_(<2 x s32>) = G_BUILD_VECTOR %11:_(s32), %11:_(s32)
   %13:_(<2 x s8>) = G_TRUNC %12
   %14:_(<2 x s8>) = COPY %13
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyTruncLoad = Copies[Copies.size() - 3];
  Register CopyTruncNeg1 = Copies[Copies.size() - 2];
  Register CopyTrunc7 = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  EXPECT_EQ(1u, Info.computeNumSignBits(CopyTruncLoad));
  EXPECT_EQ(8u, Info.computeNumSignBits(CopyTruncNeg1));
  EXPECT_EQ(5u, Info.computeNumSignBits(CopyTrunc7));
}

TEST_F(AMDGPUGISelMITest, TestVectorIsKnownToBeAPowerOfTwo) {

  StringRef MIRString = R"(
  %zero:_(s32) = G_CONSTANT i32 0
  %zero_splat:_(<2 x s32>) = G_BUILD_VECTOR %zero:_(s32), %zero:_(s32)
  %one:_(s32) = G_CONSTANT i32 1
  %one_splat:_(<2 x s32>) = G_BUILD_VECTOR %one:_(s32), %one:_(s32)
  %two:_(s32) = G_CONSTANT i32 2
  %two_splat:_(<2 x s32>) = G_BUILD_VECTOR %two:_(s32), %two:_(s32)
  %three:_(s32) = G_CONSTANT i32 3
  %three_splat:_(<2 x s32>) = G_BUILD_VECTOR %three:_(s32), %three:_(s32)
  %five:_(s32) = G_CONSTANT i32 5
  %five_splat:_(<2 x s32>) = G_BUILD_VECTOR %five:_(s32), %five:_(s32)
  %copy_zero_splat:_(<2 x s32>) = COPY %zero_splat
  %copy_one_splat:_(<2 x s32>) = COPY %one_splat
  %copy_two_splat:_(<2 x s32>) = COPY %two_splat
  %copy_three_splat:_(<2 x s32>) = COPY %three_splat

  %trunc_two_splat:_(<2 x s1>) = G_TRUNC %two_splat
  %trunc_three_splat:_(<2 x s1>) = G_TRUNC %three_splat
  %trunc_five_splat:_(<2 x s1>) = G_TRUNC %five_splat

  %copy_trunc_two_splat:_(<2 x s1>) = COPY %trunc_two_splat
  %copy_trunc_three_splat:_(<2 x s1>) = COPY %trunc_three_splat
  %copy_trunc_five_splat:_(<2 x s1>) = COPY %trunc_five_splat

  %ptr:_(p1) = G_IMPLICIT_DEF
  %shift_amt:_(<2 x s32>) = G_LOAD %ptr :: (load (<2 x s32>), addrspace 1)

  %shl_1:_(<2 x s32>) = G_SHL %one_splat, %shift_amt
  %copy_shl_1:_(<2 x s32>) = COPY %shl_1

  %shl_2:_(<2 x s32>) = G_SHL %two_splat, %shift_amt
  %copy_shl_2:_(<2 x s32>) = COPY %shl_2

  %not_sign_mask:_(<2 x s32>) = G_LOAD %ptr :: (load (<2 x s32>), addrspace 1)
  %sign_mask:_(s32) = G_CONSTANT i32 -2147483648
  %sign_mask_splat:_(<2 x s32>) = G_BUILD_VECTOR %sign_mask:_(s32), %sign_mask:_(s32)

  %lshr_not_sign_mask:_(<2 x s32>) = G_LSHR %not_sign_mask, %shift_amt
  %copy_lshr_not_sign_mask:_(<2 x s32>) = COPY %lshr_not_sign_mask

  %lshr_sign_mask:_(<2 x s32>) = G_LSHR %sign_mask_splat, %shift_amt
  %copy_lshr_sign_mask:_(<2 x s32>) = COPY %lshr_sign_mask

  %or_pow2:_(<2 x s32>) = G_OR %zero_splat, %two_splat
  %copy_or_pow2:_(<2 x s32>) = COPY %or_pow2
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  GISelKnownBits KB(*MF);

  Register CopyZero = Copies[Copies.size() - 12];
  Register CopyOne = Copies[Copies.size() - 11];
  Register CopyTwo = Copies[Copies.size() - 10];
  Register CopyThree = Copies[Copies.size() - 9];
  Register CopyTruncTwo = Copies[Copies.size() - 8];
  Register CopyTruncThree = Copies[Copies.size() - 7];
  Register CopyTruncFive = Copies[Copies.size() - 6];

  Register CopyShl1 = Copies[Copies.size() - 5];
  Register CopyShl2 = Copies[Copies.size() - 4];

  Register CopyLShrNotSignMask = Copies[Copies.size() - 3];
  Register CopyLShrSignMask = Copies[Copies.size() - 2];
  Register CopyOrPow2 = Copies[Copies.size() - 1];

  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyZero, *MRI, &KB));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyOne, *MRI, &KB));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyTwo, *MRI, &KB));
  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyThree, *MRI, &KB));

  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyTruncTwo, *MRI, &KB));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyTruncThree, *MRI, &KB));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyTruncFive, *MRI, &KB));
  // TODO: check for vector(splat) shift amount.
  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyShl1, *MRI, &KB));
  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyShl2, *MRI, &KB));

  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyLShrNotSignMask, *MRI, &KB));
  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyLShrSignMask, *MRI, &KB));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyOrPow2, *MRI, &KB));
}

TEST_F(AArch64GISelMITest, TestVectorMetadata) {
  StringRef MIRString = R"(
   %imp:_(p0) = G_IMPLICIT_DEF
   %load:_(<2 x s8>) = G_LOAD %imp(p0) :: (load (<2 x s8>))
   %ext:_(<2 x s32>) = G_ZEXT %load(<2 x s8>)
   %cst_elt:_(s32) = G_CONSTANT i32 1
   %cst:_(<2 x s32>) = G_BUILD_VECTOR %cst_elt:_(s32), %cst_elt:_(s32)
   %and:_(<2 x s32>) = G_AND %ext, %cst
   %copy:_(<2 x s32>) = COPY %and(<2 x s32>)
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  MachineInstr *And = MRI->getVRegDef(SrcReg);
  MachineInstr *Ext = MRI->getVRegDef(And->getOperand(1).getReg());
  MachineInstr *Load = MRI->getVRegDef(Ext->getOperand(1).getReg());
  IntegerType *Int8Ty = Type::getInt8Ty(Context);

  Metadata *LowAndHigh[] = {
      ConstantAsMetadata::get(ConstantInt::get(Int8Ty, 0)),
      ConstantAsMetadata::get(ConstantInt::get(Int8Ty, 2))};
  auto *NewMDNode = MDNode::get(Context, LowAndHigh);
  const MachineMemOperand *OldMMO = *Load->memoperands_begin();
  MachineMemOperand NewMMO(OldMMO->getPointerInfo(), OldMMO->getFlags(),
                           OldMMO->getMemoryType(), OldMMO->getAlign(),
                           OldMMO->getAAInfo(), NewMDNode);
  MachineIRBuilder MIB(*Load);
  MIB.buildLoad(Load->getOperand(0), Load->getOperand(1), NewMMO);
  Load->eraseFromParent();

  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(And->getOperand(1).getReg());

  EXPECT_TRUE(Res.One.isZero());

  APInt Mask(Res.getBitWidth(), 1);
  Mask.flipAllBits();
  EXPECT_EQ(Mask.getZExtValue(), Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestVectorKnownBitsExt) {
  StringRef MIRString = R"(
   %c1:_(s16) = G_CONSTANT i16 1
   %c1_splat:_(<2 x s16>) = G_BUILD_VECTOR %c1:_(s16), %c1:_(s16)
   %x:_(<2 x s16>) = G_IMPLICIT_DEF
   %y:_(<2 x s16>) = G_AND %x, %c1_splat
   %anyext:_(<2 x s32>) = G_ANYEXT %y(<2 x s16>)
   %r1:_(<2 x s32>) = COPY %anyext
   %zext:_(<2 x s32>) = G_ZEXT %y(<2 x s16>)
   %r2:_(<2 x s32>) = COPY %zext
   %sext:_(<2 x s32>) = G_SEXT %y(<2 x s16>)
   %r3:_(<2 x s32>) = COPY %sext
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  Register CopyRegAny = Copies[Copies.size() - 3];
  Register CopyRegZ = Copies[Copies.size() - 2];
  Register CopyRegS = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  MachineInstr *Copy;
  Register SrcReg;
  KnownBits Res;

  Copy = MRI->getVRegDef(CopyRegAny);
  SrcReg = Copy->getOperand(1).getReg();
  Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)32, Res.getBitWidth());
  EXPECT_EQ((uint64_t)0, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0x0000fffe, Res.Zero.getZExtValue());

  Copy = MRI->getVRegDef(CopyRegZ);
  SrcReg = Copy->getOperand(1).getReg();
  Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)32, Res.getBitWidth());
  EXPECT_EQ((uint64_t)0, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0xfffffffe, Res.Zero.getZExtValue());

  Copy = MRI->getVRegDef(CopyRegS);
  SrcReg = Copy->getOperand(1).getReg();
  Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)32, Res.getBitWidth());
  EXPECT_EQ((uint64_t)0, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0xfffffffe, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorSextInReg) {
  StringRef MIRString = R"(
   ; 000...0001
   %one:_(s32) = G_CONSTANT i32 1
   %one_splat:_(<2 x s32>) = G_BUILD_VECTOR %one:_(s32), %one:_(s32)

   ; 000...0010
   %two:_(s32) = G_CONSTANT i32 2
   %two_splat:_(<2 x s32>) = G_BUILD_VECTOR %two:_(s32), %two:_(s32)

   ; 000...1010
   %ten:_(s32) = G_CONSTANT i32 10
   %ten_splat:_(<2 x s32>) = G_BUILD_VECTOR %ten:_(s32), %ten:_(s32)

   ; ???...????
   %x0:_(<2 x s32>) = COPY $x0

   ; ???...?1?
   %or:_(<2 x s32>) = G_OR %x0, %two_splat

   ; All bits are known.
   %inreg1:_(<2 x s32>) = G_SEXT_INREG %one_splat, 1
   %copy_inreg1:_(<2 x s32>) = COPY %inreg1

   ; All bits unknown
   %inreg2:_(<2 x s32>) = G_SEXT_INREG %or, 1
   %copy_inreg2:_(<2 x s32>) = COPY %inreg2

   ; Extending from the only (known) set bit
   ; 111...11?
   %inreg3:_(<2 x s32>) = G_SEXT_INREG %or, 2
   %copy_inreg3:_(<2 x s32>) = COPY %inreg3

   ; Extending from a known set bit, overwriting all of the high set bits.
   ; 111...1110
   %inreg4:_(<2 x s32>) = G_SEXT_INREG %ten_splat, 2
   %copy_inreg4:_(<2 x s32>) = COPY %inreg4

)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  GISelKnownBits Info(*MF);
  KnownBits Res;
  auto GetKB = [&](unsigned Idx) {
    Register CopyReg = Copies[Idx];
    auto *Copy = MRI->getVRegDef(CopyReg);
    return Info.getKnownBits(Copy->getOperand(1).getReg());
  };

  Res = GetKB(Copies.size() - 4);
  EXPECT_EQ(32u, Res.getBitWidth());
  EXPECT_TRUE(Res.isAllOnes());

  Res = GetKB(Copies.size() - 3);
  EXPECT_EQ(32u, Res.getBitWidth());
  EXPECT_TRUE(Res.isUnknown());

  Res = GetKB(Copies.size() - 2);
  EXPECT_EQ(32u, Res.getBitWidth());
  EXPECT_EQ(0xFFFFFFFEu, Res.One.getZExtValue());
  EXPECT_EQ(0u, Res.Zero.getZExtValue());

  Res = GetKB(Copies.size() - 1);
  EXPECT_EQ(32u, Res.getBitWidth());
  EXPECT_EQ(0xFFFFFFFEu, Res.One.getZExtValue());
  EXPECT_EQ(1u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorAssertSext) {
  StringRef MIRString = R"(
   ; 000...0001
   %one:_(s32) = G_CONSTANT i32 1
   %one_splat:_(<2 x s32>) = G_BUILD_VECTOR %one, %one

   ; 000...0010
   %two:_(s32) = G_CONSTANT i32 2
   %two_splat:_(<2 x s32>) = G_BUILD_VECTOR %two, %two

   ; 000...1010
   %ten:_(s32) = G_CONSTANT i32 10
   %ten_splat:_(<2 x s32>) = G_BUILD_VECTOR %ten, %ten

   ; ???...????
   %x0:_(<2 x s32>) = COPY $x0

   ; ???...?1?
   %or:_(<2 x s32>) = G_OR %x0, %two_splat

   ; All bits are known.
   %assert_sext1:_(<2 x s32>) = G_ASSERT_SEXT %one_splat, 1
   %copy_assert_sext1:_(<2 x s32>) = COPY %assert_sext1

   ; All bits unknown
   %assert_sext2:_(<2 x s32>) = G_ASSERT_SEXT %or, 1
   %copy_assert_sext2:_(<2 x s32>) = COPY %assert_sext2

   ; Extending from the only (known) set bit
   ; 111...11?
   %assert_sext3:_(<2 x s32>) = G_ASSERT_SEXT %or, 2
   %copy_assert_sext3:_(<2 x s32>) = COPY %assert_sext3

   ; Extending from a known set bit, overwriting all of the high set bits.
   ; 111...1110
   %assert_sext4:_(<2 x s32>) = G_ASSERT_SEXT %ten_splat, 2
   %copy_assert_sext4:_(<2 x s32>) = COPY %assert_sext4
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();
  GISelKnownBits Info(*MF);
  KnownBits Res;
  auto GetKB = [&](unsigned Idx) {
    Register CopyReg = Copies[Idx];
    auto *Copy = MRI->getVRegDef(CopyReg);
    return Info.getKnownBits(Copy->getOperand(1).getReg());
  };

  // Every bit is known to be a 1.
  Res = GetKB(Copies.size() - 4);
  EXPECT_EQ(32u, Res.getBitWidth());
  EXPECT_TRUE(Res.isAllOnes());

  // All bits are unknown
  Res = GetKB(Copies.size() - 3);
  EXPECT_EQ(32u, Res.getBitWidth());
  EXPECT_TRUE(Res.isUnknown());

  // Extending from the only known set bit
  // 111...11?
  Res = GetKB(Copies.size() - 2);
  EXPECT_EQ(32u, Res.getBitWidth());
  EXPECT_EQ(0xFFFFFFFEu, Res.One.getZExtValue());
  EXPECT_EQ(0u, Res.Zero.getZExtValue());

  // Extending from a known set bit, overwriting all of the high set bits.
  // 111...1110
  Res = GetKB(Copies.size() - 1);
  EXPECT_EQ(32u, Res.getBitWidth());
  EXPECT_EQ(0xFFFFFFFEu, Res.One.getZExtValue());
  EXPECT_EQ(1u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestVectorKnownBitsBSwapBitReverse) {
  StringRef MIRString = R"(
   %const:_(s32) = G_CONSTANT i32 287454020
   %const_splat:_(<2 x s32>) = G_BUILD_VECTOR %const:_(s32), %const:_(s32)
   %bswap:_(<2 x s32>) = G_BSWAP %const_splat
   %bitreverse:_(<2 x s32>) = G_BITREVERSE %const_splat
   %copy_bswap:_(<2 x s32>) = COPY %bswap
   %copy_bitreverse:_(<2 x s32>) = COPY %bitreverse
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  const uint32_t ByteSwappedVal = 0x44332211;
  const uint32_t BitSwappedVal = 0x22cc4488;

  Register CopyBSwap = Copies[Copies.size() - 2];
  Register CopyBitReverse = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);

  KnownBits BSwapKnown = Info.getKnownBits(CopyBSwap);
  EXPECT_EQ(32u, BSwapKnown.getBitWidth());
  EXPECT_EQ(ByteSwappedVal, BSwapKnown.One.getZExtValue());
  EXPECT_EQ(~ByteSwappedVal, BSwapKnown.Zero.getZExtValue());

  KnownBits BitReverseKnown = Info.getKnownBits(CopyBitReverse);
  EXPECT_EQ(32u, BitReverseKnown.getBitWidth());
  EXPECT_EQ(BitSwappedVal, BitReverseKnown.One.getZExtValue());
  EXPECT_EQ(~BitSwappedVal, BitReverseKnown.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorUMAX) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 10
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 1
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s8) = G_CONSTANT i8 3
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 12
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %umax0:_(<2 x s8>) = G_UMAX %val0, %val1
   %copy_umax0:_(<2 x s8>) = COPY %umax0

   %mask4:_(s8) = G_CONSTANT i8 14
   %mask4_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask4, %mask4
   %mask5:_(s8) = G_CONSTANT i8 2
   %mask5_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask5, %mask5
   %tmp3:_(<2 x s8>) = G_AND %unknown, %mask4_splat
   %val3:_(<2 x s8>) = G_OR %tmp3, %mask5_splat
   %mask6:_(s8) = G_CONSTANT i8 4
   %mask6_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask6, %mask6
   %mask7:_(s8) = G_CONSTANT i8 11
   %mask7_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask7, %mask7
   %tmp4:_(<2 x s8>) = G_AND %unknown, %mask6_splat
   %val4:_(<2 x s8>) = G_OR %tmp4, %mask7_splat
   %umax1:_(<2 x s8>) = G_UMAX %val3, %val4
   %copy_umax1:_(<2 x s8>) = COPY %umax1
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg0 = Copies[Copies.size() - 2];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  EXPECT_EQ(0x0Cu, Res0.One.getZExtValue());
  EXPECT_EQ(0xF0u, Res0.Zero.getZExtValue());

  Register CopyReg1 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy1 = MRI->getVRegDef(CopyReg1);
  Register SrcReg1 = FinalCopy1->getOperand(1).getReg();
  KnownBits Res1 = Info.getKnownBits(SrcReg1);
  EXPECT_EQ(0x0Au, Res1.One.getZExtValue());
  EXPECT_EQ(0xF0u, Res1.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestVectorKnownBitsUMax) {
  StringRef MIRString = R"(
   %val:_(<2 x s32>) = COPY $x0
   %zext:_(<2 x s64>) = G_ZEXT %val
   %const:_(s64) = G_CONSTANT i64 -256
   %const_splat:_(<2 x s64>) = G_BUILD_VECTOR %const:_(s64), %const:_(s64)
   %umax:_(<2 x s64>) = G_UMAX %zext, %const_splat
   %copy_umax:_(<2 x s64>) = COPY %umax
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyUMax = Copies[Copies.size() - 1];
  GISelKnownBits Info(*MF);

  KnownBits KnownUmax = Info.getKnownBits(CopyUMax);
  EXPECT_EQ(64u, KnownUmax.getBitWidth());
  EXPECT_EQ(0xffu, KnownUmax.Zero.getZExtValue());
  EXPECT_EQ(0xffffffffffffff00, KnownUmax.One.getZExtValue());

  EXPECT_EQ(0xffu, KnownUmax.Zero.getZExtValue());
  EXPECT_EQ(0xffffffffffffff00, KnownUmax.One.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorUMIN) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 10
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 1
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s8) = G_CONSTANT i8 3
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 12
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %umin:_(<2 x s8>) = G_UMIN %val0, %val1
   %copy_umin:_(<2 x s8>) = COPY %umin
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg0 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  EXPECT_EQ(0x01u, Res0.One.getZExtValue());
  EXPECT_EQ(0xF4u, Res0.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorSMAX) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 128
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 64
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s8) = G_CONSTANT i8 1
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 128
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %smax:_(<2 x s8>) = G_SMAX %val0, %val1
   %copy_smax:_(<2 x s8>) = COPY %smax
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg0 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  EXPECT_EQ(0x40u, Res0.One.getZExtValue());
  EXPECT_EQ(0x3Fu, Res0.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorSMIN) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(<2 x s8>) = G_LOAD %ptr(p0) :: (load (<2 x s8>))
   %mask0:_(s8) = G_CONSTANT i8 128
   %mask0_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask0, %mask0
   %mask1:_(s8) = G_CONSTANT i8 64
   %mask1_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask1, %mask1
   %tmp0:_(<2 x s8>) = G_AND %unknown, %mask0_splat
   %val0:_(<2 x s8>) = G_OR %tmp0, %mask1_splat
   %mask2:_(s8) = G_CONSTANT i8 1
   %mask2_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask2, %mask2
   %mask3:_(s8) = G_CONSTANT i8 128
   %mask3_splat:_(<2 x s8>) = G_BUILD_VECTOR %mask3, %mask3
   %tmp1:_(<2 x s8>) = G_AND %unknown, %mask2_splat
   %val1:_(<2 x s8>) = G_OR %tmp1, %mask3_splat
   %smin:_(<2 x s8>) = G_SMIN %val0, %val1
   %copy_smin:_(<2 x s8>) = COPY %smin
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg0 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  EXPECT_EQ(0x80u, Res0.One.getZExtValue());
  EXPECT_EQ(0x7Eu, Res0.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestVectorInvalidQueries) {
  StringRef MIRString = R"(
   %src:_(<2 x s32>) = COPY $x0
   %thirty2:_(s32) = G_CONSTANT i32 32
   %thirty2_splat:_(<2 x s32>) = G_BUILD_VECTOR %thirty2:_(s32), %thirty2:_(s32)
   %equalSized:_(<2 x s32>) = G_SHL %src, %thirty2_splat
   %copy1:_(<2 x s32>) = COPY %equalSized
   %thirty3:_(s32) = G_CONSTANT i32 33
   %thirty3_splat:_(<2 x s32>) = G_BUILD_VECTOR %thirty3:_(s32), %thirty3:_(s32)
   %biggerSized:_(<2 x s32>) = G_SHL %src, %thirty3_splat
   %copy2:_(<2 x s32>) = COPY %biggerSized
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register EqSizedCopyReg = Copies[Copies.size() - 2];
  MachineInstr *EqSizedCopy = MRI->getVRegDef(EqSizedCopyReg);
  Register EqSizedShl = EqSizedCopy->getOperand(1).getReg();

  Register BiggerSizedCopyReg = Copies[Copies.size() - 1];
  MachineInstr *BiggerSizedCopy = MRI->getVRegDef(BiggerSizedCopyReg);
  Register BiggerSizedShl = BiggerSizedCopy->getOperand(1).getReg();

  GISelKnownBits Info(*MF);
  KnownBits EqSizeRes = Info.getKnownBits(EqSizedShl);
  KnownBits BiggerSizeRes = Info.getKnownBits(BiggerSizedShl);

  EXPECT_TRUE(EqSizeRes.One.isZero());
  EXPECT_TRUE(EqSizeRes.Zero.isAllOnes());

  EXPECT_TRUE(BiggerSizeRes.One.isZero());
  EXPECT_TRUE(BiggerSizeRes.Zero.isAllOnes());
}

TEST_F(AArch64GISelMITest, TestKnownBitsVectorAssertZext) {
  StringRef MIRString = R"(
   %copy_x0:_(s64) = COPY $x0
   %copy_x1:_(s64) = COPY $x1
   %x0_x1:_(<2 x s64>) = G_BUILD_VECTOR %copy_x0, %copy_x1

   %assert8:_(<2 x s64>) = G_ASSERT_ZEXT %x0_x1, 8
   %copy_assert8:_(<2 x s64>) = COPY %assert8

   %assert1:_(<2 x s64>) = G_ASSERT_ZEXT %x0_x1, 1
   %copy_assert1:_(<2 x s64>) = COPY %assert1

   %assert63:_(<2 x s64>) = G_ASSERT_ZEXT %x0_x1, 63
   %copy_assert63:_(<2 x s64>) = COPY %assert63

   %assert3:_(<2 x s64>) = G_ASSERT_ZEXT %x0_x1, 3
   %copy_assert3:_(<2 x s64>) = COPY %assert3
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyAssert8 = Copies[Copies.size() - 4];
  Register CopyAssert1 = Copies[Copies.size() - 3];
  Register CopyAssert63 = Copies[Copies.size() - 2];
  Register CopyAssert3 = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  MachineInstr *Copy;
  Register SrcReg;
  KnownBits Res;

  // Assert zero-extension from an 8-bit value.
  Copy = MRI->getVRegDef(CopyAssert8);
  SrcReg = Copy->getOperand(1).getReg();
  Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(64u, Res.getBitWidth());
  EXPECT_EQ(0u, Res.One.getZExtValue());
  EXPECT_EQ(0xFFFFFFFFFFFFFF00u, Res.Zero.getZExtValue());

  // Assert zero-extension from a 1-bit value.
  Copy = MRI->getVRegDef(CopyAssert1);
  SrcReg = Copy->getOperand(1).getReg();
  Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(64u, Res.getBitWidth());
  EXPECT_EQ(0u, Res.One.getZExtValue());
  EXPECT_EQ(0xFFFFFFFFFFFFFFFE, Res.Zero.getZExtValue());

  // Assert zero-extension from a 63-bit value.
  Copy = MRI->getVRegDef(CopyAssert63);
  SrcReg = Copy->getOperand(1).getReg();
  Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(64u, Res.getBitWidth());
  EXPECT_EQ(0u, Res.One.getZExtValue());
  EXPECT_EQ(0x8000000000000000u, Res.Zero.getZExtValue());

  // Assert zero-extension from a 3-bit value.
  Copy = MRI->getVRegDef(CopyAssert3);
  SrcReg = Copy->getOperand(1).getReg();
  Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(64u, Res.getBitWidth());
  EXPECT_EQ(0u, Res.One.getZExtValue());
  EXPECT_EQ(0xFFFFFFFFFFFFFFF8u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestNumSignBitsUAddoOverflow) {
  StringRef MIRString = R"(
   %copy_x0:_(s64) = COPY $x0
   %copy_x1:_(s64) = COPY $x1
   %x0_x1:_(<2 x s64>) = G_BUILD_VECTOR %copy_x0, %copy_x1
   %uaddo:_(<2 x s64>), %overflow:_(<2 x s32>) = G_UADDO %x0_x1, %x0_x1
   %result:_(<2 x s32>) = COPY %overflow
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyOverflow = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);

  // Assert sign-extension from vector boolean
  EXPECT_EQ(32u, Info.computeNumSignBits(CopyOverflow));
}

TEST_F(AArch64GISelMITest, TestKnwonBitsUnmergeVectorScalar) {
  StringRef MIRString = R"(
   %copy_x0:_(<2 x s16>) = COPY $w0
   %maskf:_(s16) = G_CONSTANT i16 15
   %x0_x1:_(<2 x s16>) = G_BUILD_VECTOR %maskf, %maskf
   %and:_(<2 x s16>) = G_AND %copy_x0, %x0_x1
   %x0_0:_(s16), %x0_1:_(s16) = G_UNMERGE_VALUES %and
   %result:_(s16) = COPY %x0_0
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyOverflow = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);

  EXPECT_EQ(0xFFF0u, Info.getKnownBits(CopyOverflow).Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnwonBitsUnmergeVectorVector) {
  StringRef MIRString = R"(
   %copy_x0:_(<4 x s8>) = COPY $w0
   %maskff:_(s8) = G_CONSTANT i8 255
   %maskf:_(s8) = G_CONSTANT i8 15
   %x0_x1:_(<4 x s8>) = G_BUILD_VECTOR %maskf, %maskf, %maskff, %maskff
   %and:_(<4 x s8>) = G_AND %copy_x0, %x0_x1
   %x0_0:_(<2 x s8>), %x0_1:_(<2 x s8>) = G_UNMERGE_VALUES %and
   %result1:_(<2 x s8>) = COPY %x0_0
   %result2:_(<2 x s8>) = COPY %x0_1
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  GISelKnownBits Info(*MF);

  Register CopyOverflow1 = Copies[Copies.size() - 2];
  EXPECT_EQ(0xF0u, Info.getKnownBits(CopyOverflow1).Zero.getZExtValue());
  Register CopyOverflow2 = Copies[Copies.size() - 1];
  EXPECT_EQ(0x00u, Info.getKnownBits(CopyOverflow2).Zero.getZExtValue());
}
