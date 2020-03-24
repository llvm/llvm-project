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

TEST_F(AArch64GISelMITest, TestKnownBitsCst) {
  StringRef MIRString = "  %3:_(s8) = G_CONSTANT i8 1\n"
                        "  %4:_(s8) = COPY %3\n";
  setUp(MIRString);
  if (!TM)
    return;
  unsigned CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  unsigned SrcReg = FinalCopy->getOperand(1).getReg();
  unsigned DstReg = FinalCopy->getOperand(0).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)1, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0xfe, Res.Zero.getZExtValue());

  KnownBits Res2 = Info.getKnownBits(DstReg);
  EXPECT_EQ(Res.One.getZExtValue(), Res2.One.getZExtValue());
  EXPECT_EQ(Res.Zero.getZExtValue(), Res2.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsCstWithClass) {
  StringRef MIRString = "  %10:gpr32 = MOVi32imm 1\n"
                        "  %4:_(s32) = COPY %10\n";
  setUp(MIRString);
  if (!TM)
    return;
  unsigned CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  unsigned SrcReg = FinalCopy->getOperand(1).getReg();
  unsigned DstReg = FinalCopy->getOperand(0).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  // We can't analyze %3 due to the register class constraint. We will get a
  // default-constructed KnownBits back.
  EXPECT_EQ((uint64_t)1, Res.getBitWidth());
  EXPECT_EQ((uint64_t)0, Res.One.getZExtValue());
  EXPECT_EQ((uint64_t)0, Res.Zero.getZExtValue());

  KnownBits Res2 = Info.getKnownBits(DstReg);
  // We still don't know the values due to the register class constraint but %4
  // did reveal the size of %3.
  EXPECT_EQ((uint64_t)32, Res2.getBitWidth());
  EXPECT_EQ(Res.One.getZExtValue(), Res2.One.getZExtValue());
  EXPECT_EQ(Res.Zero.getZExtValue(), Res2.Zero.getZExtValue());
}

// Check that we are able to track bits through PHIs
// and get the intersections of everything we know on each operand.
TEST_F(AArch64GISelMITest, TestKnownBitsCstPHI) {
  StringRef MIRString = "  bb.10:\n"
                        "  %10:_(s8) = G_CONSTANT i8 3\n"
                        "  %11:_(s1) = G_IMPLICIT_DEF\n"
                        "  G_BRCOND %11(s1), %bb.11\n"
                        "  G_BR %bb.12\n"
                        "\n"
                        "  bb.11:\n"
                        "  %12:_(s8) = G_CONSTANT i8 2\n"
                        "  G_BR %bb.12\n"
                        "\n"
                        "  bb.12:\n"
                        "  %13:_(s8) = PHI %10(s8), %bb.10, %12(s8), %bb.11\n"
                        "  %14:_(s8) = COPY %13\n";
  setUp(MIRString);
  if (!TM)
    return;
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

// Check that we report we know nothing when we hit a
// non-generic register.
// Note: this could be improved though!
TEST_F(AArch64GISelMITest, TestKnownBitsCstPHIToNonGenericReg) {
  StringRef MIRString = "  bb.10:\n"
                        "  %10:gpr32 = MOVi32imm 3\n"
                        "  %11:_(s1) = G_IMPLICIT_DEF\n"
                        "  G_BRCOND %11(s1), %bb.11\n"
                        "  G_BR %bb.12\n"
                        "\n"
                        "  bb.11:\n"
                        "  %12:_(s8) = G_CONSTANT i8 2\n"
                        "  G_BR %bb.12\n"
                        "\n"
                        "  bb.12:\n"
                        "  %13:_(s8) = PHI %10, %bb.10, %12(s8), %bb.11\n"
                        "  %14:_(s8) = COPY %13\n";
  setUp(MIRString);
  if (!TM)
    return;
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

// Check that we know nothing when at least one value of a PHI
// comes from something we cannot analysis.
// This test is not particularly interesting, it is just
// here to cover the code that stops the analysis of PHIs
// earlier. In that case, we would not even look at the
// second incoming value.
TEST_F(AArch64GISelMITest, TestKnownBitsUnknownPHI) {
  StringRef MIRString =
      "  bb.10:\n"
      "  %10:_(s64) = COPY %0\n"
      "  %11:_(s1) = G_IMPLICIT_DEF\n"
      "  G_BRCOND %11(s1), %bb.11\n"
      "  G_BR %bb.12\n"
      "\n"
      "  bb.11:\n"
      "  %12:_(s64) = G_CONSTANT i64 2\n"
      "  G_BR %bb.12\n"
      "\n"
      "  bb.12:\n"
      "  %13:_(s64) = PHI %10(s64), %bb.10, %12(s64), %bb.11\n"
      "  %14:_(s64) = COPY %13\n";
  setUp(MIRString);
  if (!TM)
    return;
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

// Check that we manage to process PHIs that loop on themselves.
// For now, the analysis just stops and assumes it knows nothing,
// eventually we could teach it how to properly track phis that
// loop back.
TEST_F(AArch64GISelMITest, TestKnownBitsCstPHIWithLoop) {
  StringRef MIRString =
      "  bb.10:\n"
      "  %10:_(s8) = G_CONSTANT i8 3\n"
      "  %11:_(s1) = G_IMPLICIT_DEF\n"
      "  G_BRCOND %11(s1), %bb.11\n"
      "  G_BR %bb.12\n"
      "\n"
      "  bb.11:\n"
      "  %12:_(s8) = G_CONSTANT i8 2\n"
      "  G_BR %bb.12\n"
      "\n"
      "  bb.12:\n"
      "  %13:_(s8) = PHI %10(s8), %bb.10, %12(s8), %bb.11, %14(s8), %bb.12\n"
      "  %14:_(s8) = COPY %13\n"
      "  G_BR %bb.12\n";
  setUp(MIRString);
  if (!TM)
    return;
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

// Check that we don't try to analysis PHIs progression.
// Setting a deep enough max depth would allow to effectively simulate
// what happens in the loop.
// Thus, with a deep enough depth, we could actually figure out
// that %14's zero known bits are actually at least what we know
// for %10, right shifted by one.
// However, this process is super expensive compile-time wise and
// we don't want to reach that conclusion while playing with max depth.
// For now, the analysis just stops and assumes it knows nothing
// on PHIs, but eventually we could teach it how to properly track
// phis that loop back without relying on the luck effect of max
// depth.
TEST_F(AArch64GISelMITest, TestKnownBitsDecreasingCstPHIWithLoop) {
  StringRef MIRString = "  bb.10:\n"
                        "  %10:_(s8) = G_CONSTANT i8 5\n"
                        "  %11:_(s8) = G_CONSTANT i8 1\n"
                        "\n"
                        "  bb.12:\n"
                        "  %13:_(s8) = PHI %10(s8), %bb.10, %14(s8), %bb.12\n"
                        "  %14:_(s8) = G_LSHR %13, %11\n"
                        "  %15:_(s8) = COPY %14\n"
                        "  G_BR %bb.12\n";
  setUp(MIRString);
  if (!TM)
    return;
  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  Register DstReg = FinalCopy->getOperand(0).getReg();
  GISelKnownBits Info(*MF, /*MaxDepth=*/24);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ((uint64_t)0, Res.One.getZExtValue());
  // A single iteration on the PHI (%13) gives:
  // %10 has known zero of 0xFA
  // %12 has known zero of 0x80 (we shift right by one so high bit is zero)
  // Therefore, %14's known zero are 0x80 shifted by one 0xC0.
  // If we had simulated the loop we could have more zero bits, basically
  // up to 0xFC (count leading zero of 5, + 1).
  EXPECT_EQ((uint64_t)0xC0, Res.Zero.getZExtValue());

  KnownBits Res2 = Info.getKnownBits(DstReg);
  EXPECT_EQ(Res.One.getZExtValue(), Res2.One.getZExtValue());
  EXPECT_EQ(Res.Zero.getZExtValue(), Res2.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsPtrToIntViceVersa) {
  StringRef MIRString = "  %3:_(s16) = G_CONSTANT i16 256\n"
                        "  %4:_(p0) = G_INTTOPTR %3\n"
                        "  %5:_(s32) = G_PTRTOINT %4\n"
                        "  %6:_(s32) = COPY %5\n";
  setUp(MIRString);
  if (!TM)
    return;
  unsigned CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  unsigned SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(256u, Res.One.getZExtValue());
  EXPECT_EQ(0xfffffeffu, Res.Zero.getZExtValue());
}
TEST_F(AArch64GISelMITest, TestKnownBitsXOR) {
  StringRef MIRString = "  %3:_(s8) = G_CONSTANT i8 4\n"
                        "  %4:_(s8) = G_CONSTANT i8 7\n"
                        "  %5:_(s8) = G_XOR %3, %4\n"
                        "  %6:_(s8) = COPY %5\n";
  setUp(MIRString);
  if (!TM)
    return;
  unsigned CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  unsigned SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  EXPECT_EQ(3u, Res.One.getZExtValue());
  EXPECT_EQ(252u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBits) {

  StringRef MIR = "  %3:_(s32) = G_TRUNC %0\n"
                  "  %4:_(s32) = G_TRUNC %1\n"
                  "  %5:_(s32) = G_CONSTANT i32 5\n"
                  "  %6:_(s32) = G_CONSTANT i32 24\n"
                  "  %7:_(s32) = G_CONSTANT i32 28\n"
                  "  %14:_(p0) = G_INTTOPTR %7\n"
                  "  %16:_(s32) = G_PTRTOINT %14\n"
                  "  %8:_(s32) = G_SHL %3, %5\n"
                  "  %9:_(s32) = G_SHL %4, %5\n"
                  "  %10:_(s32) = G_OR %8, %6\n"
                  "  %11:_(s32) = G_OR %9, %16\n"
                  "  %12:_(s32) = G_MUL %10, %11\n"
                  "  %13:_(s32) = COPY %12\n";
  setUp(MIR);
  if (!TM)
    return;
  unsigned CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  unsigned SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Known = Info.getKnownBits(SrcReg);
  EXPECT_FALSE(Known.hasConflict());
  EXPECT_EQ(0u, Known.One.getZExtValue());
  EXPECT_EQ(31u, Known.Zero.getZExtValue());
  APInt Zeroes = Info.getKnownZeroes(SrcReg);
  EXPECT_EQ(Known.Zero, Zeroes);
}

TEST_F(AArch64GISelMITest, TestSignBitIsZero) {
  setUp();
  if (!TM)
    return;

  const LLT S32 = LLT::scalar(32);
  auto SignBit = B.buildConstant(S32, 0x80000000);
  auto Zero = B.buildConstant(S32, 0);

  GISelKnownBits KnownBits(*MF);

  EXPECT_TRUE(KnownBits.signBitIsZero(Zero.getReg(0)));
  EXPECT_FALSE(KnownBits.signBitIsZero(SignBit.getReg(0)));
}

TEST_F(AArch64GISelMITest, TestNumSignBitsConstant) {
  StringRef MIRString = "  %3:_(s8) = G_CONSTANT i8 1\n"
                        "  %4:_(s8) = COPY %3\n"

                        "  %5:_(s8) = G_CONSTANT i8 -1\n"
                        "  %6:_(s8) = COPY %5\n"

                        "  %7:_(s8) = G_CONSTANT i8 127\n"
                        "  %8:_(s8) = COPY %7\n"

                        "  %9:_(s8) = G_CONSTANT i8 32\n"
                        "  %10:_(s8) = COPY %9\n"

                        "  %11:_(s8) = G_CONSTANT i8 -32\n"
                        "  %12:_(s8) = COPY %11\n";
  setUp(MIRString);
  if (!TM)
    return;
  Register CopyReg1 = Copies[Copies.size() - 5];
  Register CopyRegNeg1 = Copies[Copies.size() - 4];
  Register CopyReg127 = Copies[Copies.size() - 3];
  Register CopyReg32 = Copies[Copies.size() - 2];
  Register CopyRegNeg32 = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  EXPECT_EQ(7u, Info.computeNumSignBits(CopyReg1));
  EXPECT_EQ(8u, Info.computeNumSignBits(CopyRegNeg1));
  EXPECT_EQ(1u, Info.computeNumSignBits(CopyReg127));
  EXPECT_EQ(2u, Info.computeNumSignBits(CopyReg32));
  EXPECT_EQ(3u, Info.computeNumSignBits(CopyRegNeg32));
}

TEST_F(AArch64GISelMITest, TestNumSignBitsSext) {
  StringRef MIRString = "  %3:_(p0) = G_IMPLICIT_DEF\n"
                        "  %4:_(s8) = G_LOAD %3 :: (load 1)\n"
                        "  %5:_(s32) = G_SEXT %4\n"
                        "  %6:_(s32) = COPY %5\n"

                        "  %7:_(s8) = G_CONSTANT i8 -1\n"
                        "  %8:_(s32) = G_SEXT %7\n"
                        "  %9:_(s32) = COPY %8\n";
  setUp(MIRString);
  if (!TM)
    return;
  Register CopySextLoad = Copies[Copies.size() - 2];
  Register CopySextNeg1 = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  EXPECT_EQ(25u, Info.computeNumSignBits(CopySextLoad));
  EXPECT_EQ(32u, Info.computeNumSignBits(CopySextNeg1));
}

TEST_F(AArch64GISelMITest, TestNumSignBitsTrunc) {
  StringRef MIRString = "  %3:_(p0) = G_IMPLICIT_DEF\n"
                        "  %4:_(s32) = G_LOAD %3 :: (load 4)\n"
                        "  %5:_(s8) = G_TRUNC %4\n"
                        "  %6:_(s8) = COPY %5\n"

                        "  %7:_(s32) = G_CONSTANT i32 -1\n"
                        "  %8:_(s8) = G_TRUNC %7\n"
                        "  %9:_(s8) = COPY %8\n"

                        "  %10:_(s32) = G_CONSTANT i32 7\n"
                        "  %11:_(s8) = G_TRUNC %10\n"
                        "  %12:_(s8) = COPY %11\n";
  setUp(MIRString);
  if (!TM)
    return;
  Register CopyTruncLoad = Copies[Copies.size() - 3];
  Register CopyTruncNeg1 = Copies[Copies.size() - 2];
  Register CopyTrunc7 = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  EXPECT_EQ(1u, Info.computeNumSignBits(CopyTruncLoad));
  EXPECT_EQ(8u, Info.computeNumSignBits(CopyTruncNeg1));
  EXPECT_EQ(5u, Info.computeNumSignBits(CopyTrunc7));
}
