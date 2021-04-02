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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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

TEST_F(AArch64GISelMITest, TestKnownBitsAND) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 52
   %mask1:_(s8) = G_CONSTANT i8 10
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %mask2:_(s8) = G_CONSTANT i8 32
   %mask3:_(s8) = G_CONSTANT i8 24
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %and:_(s8) = G_AND %val0, %val1
   %copy_and:_(s8) = COPY %and
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  //   00??1?10
  // & 00?11000
  // = 00??1000
  EXPECT_EQ(0x08u, Res.One.getZExtValue());
  EXPECT_EQ(0xC7u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsOR) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 52
   %mask1:_(s8) = G_CONSTANT i8 10
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %mask2:_(s8) = G_CONSTANT i8 32
   %mask3:_(s8) = G_CONSTANT i8 24
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %or:_(s8) = G_OR %val0, %val1
   %copy_or:_(s8) = COPY %or
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  //   00??1?10
  // | 00?11000
  // = 00?11?10
  EXPECT_EQ(0x1Au, Res.One.getZExtValue());
  EXPECT_EQ(0xC1u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsXOR) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 52
   %mask1:_(s8) = G_CONSTANT i8 10
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %mask2:_(s8) = G_CONSTANT i8 32
   %mask3:_(s8) = G_CONSTANT i8 24
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %xor:_(s8) = G_XOR %val0, %val1
   %copy_xor:_(s8) = COPY %xor
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  // Xor KnowBits does not track if we are doing xor of unknown bit with itself
  // or negated itself.
  //   00??1?10
  // ^ 00?11000
  // = 00??0?10
  EXPECT_EQ(0x02u, Res.One.getZExtValue());
  EXPECT_EQ(0xC9u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsXORConstant) {
  StringRef MIRString = "  %3:_(s8) = G_CONSTANT i8 4\n"
                        "  %4:_(s8) = G_CONSTANT i8 7\n"
                        "  %5:_(s8) = G_XOR %3, %4\n"
                        "  %6:_(s8) = COPY %5\n";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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

TEST_F(AArch64GISelMITest, TestKnownBitsASHR) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 38
   %mask1:_(s8) = G_CONSTANT i8 202
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %cst0:_(s8) = G_CONSTANT i8 2
   %ashr0:_(s8) = G_ASHR %val0, %cst0
   %copy_ashr0:_(s8) = COPY %ashr0

   %mask2:_(s8) = G_CONSTANT i8 204
   %mask3:_(s8) = G_CONSTANT i8 18
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %ashr1:_(s8) = G_ASHR %val1, %cst0
   %copy_ashr1:_(s8) = COPY %ashr1
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg0 = Copies[Copies.size() - 2];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  //   11?01??0 >> 2
  // = 1111?01?
  EXPECT_EQ(0xF2u, Res0.One.getZExtValue());
  EXPECT_EQ(0x04u, Res0.Zero.getZExtValue());

  Register CopyReg1 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy1 = MRI->getVRegDef(CopyReg1);
  Register SrcReg1 = FinalCopy1->getOperand(1).getReg();
  KnownBits Res1 = Info.getKnownBits(SrcReg1);
  //   ??01??10 >> 2
  // = ????01??
  EXPECT_EQ(0x04u, Res1.One.getZExtValue());
  EXPECT_EQ(0x08u, Res1.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsLSHR) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 38
   %mask1:_(s8) = G_CONSTANT i8 202
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %cst0:_(s8) = G_CONSTANT i8 2
   %lshr0:_(s8) = G_LSHR %val0, %cst0
   %copy_lshr0:_(s8) = COPY %lshr0

   %mask2:_(s8) = G_CONSTANT i8 204
   %mask3:_(s8) = G_CONSTANT i8 18
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %lshr1:_(s8) = G_LSHR %val1, %cst0
   %copy_lshr1:_(s8) = COPY %lshr1
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg0 = Copies[Copies.size() - 2];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  //   11?01??0 >> 2
  // = 0011?01?
  EXPECT_EQ(0x32u, Res0.One.getZExtValue());
  EXPECT_EQ(0xC4u, Res0.Zero.getZExtValue());

  Register CopyReg1 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy1 = MRI->getVRegDef(CopyReg1);
  Register SrcReg1 = FinalCopy1->getOperand(1).getReg();
  KnownBits Res1 = Info.getKnownBits(SrcReg1);
  //   ??01??10 >> 2
  // = 00??01??
  EXPECT_EQ(0x04u, Res1.One.getZExtValue());
  EXPECT_EQ(0xC8u, Res1.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsSHL) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 51
   %mask1:_(s8) = G_CONSTANT i8 72
   %tmp:_(s8) = G_AND %unknown, %mask0
   %val:_(s8) = G_OR %tmp, %mask1
   %cst:_(s8) = G_CONSTANT i8 3
   %shl:_(s8) = G_SHL %val, %cst
   %copy_shl:_(s8) = COPY %shl
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  //   01??10?? << 3
  // = ?10??000
  EXPECT_EQ(0x40u, Res.One.getZExtValue());
  EXPECT_EQ(0x27u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsADD) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s16) = G_LOAD %ptr(p0) :: (load 2)
   %mask0:_(s16) = G_CONSTANT i16 4642
   %mask1:_(s16) = G_CONSTANT i16 9536
   %tmp0:_(s16) = G_AND %unknown, %mask0
   %val0:_(s16) = G_OR %tmp0, %mask1
   %mask2:_(s16) = G_CONSTANT i16 4096
   %mask3:_(s16) = G_CONSTANT i16 371
   %tmp1:_(s16) = G_AND %unknown, %mask2
   %val1:_(s16) = G_OR %tmp1, %mask3
   %add:_(s16) = G_ADD %val0, %val1
   %copy_add:_(s16) = COPY %add
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  // Add KnowBits works out known carry bits first and then calculates result.
  //   001?01?101?000?0
  // + 000?000101110011
  // = 0??????01??10??1
  EXPECT_EQ(0x0091u, Res.One.getZExtValue());
  EXPECT_EQ(0x8108u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsSUB) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s16) = G_LOAD %ptr(p0) :: (load 2)
   %mask0:_(s16) = G_CONSTANT i16 4642
   %mask1:_(s16) = G_CONSTANT i16 9536
   %tmp0:_(s16) = G_AND %unknown, %mask0
   %val0:_(s16) = G_OR %tmp0, %mask1
   %mask2:_(s16) = G_CONSTANT i16 4096
   %mask3:_(s16) = G_CONSTANT i16 371
   %tmp1:_(s16) = G_AND %unknown, %mask2
   %val1:_(s16) = G_OR %tmp1, %mask3
   %sub:_(s16) = G_SUB %val0, %val1
   %copy_sub:_(s16) = COPY %sub
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  // Sub KnowBits for LHS - RHS use Add KnownBits for LHS + ~RHS + 1.
  EXPECT_EQ(0x01CDu, Res.One.getZExtValue());
  EXPECT_EQ(0xC810u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsMUL) {
  StringRef MIRString = R"(
   %ptr0:_(p0) = G_IMPLICIT_DEF
   %load0:_(s16) = G_LOAD %ptr0(p0) :: (load 2)
   %mask0:_(s16) = G_CONSTANT i16 4
   %mask1:_(s16) = G_CONSTANT i16 18
   %tmp:_(s16) = G_AND %load0, %mask0
   %val0:_(s16) = G_OR %tmp, %mask1
   %cst:_(s16) = G_CONSTANT i16 12
   %mul:_(s16) = G_MUL %val0, %cst
   %copy_mul:_(s16) = COPY %mul
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  // Mul KnowBits are conservatively correct, but not guaranteed to be precise.
  // Precise for trailing bits up to the first unknown bit.
  // 00010?10 * 00001100 =
  //          00010?1000
  //  +      00010?10000
  //  = 0000000010??1000
  // KB 0000000?????1000
  EXPECT_EQ(0x0008u, Res.One.getZExtValue());
  EXPECT_EQ(0xFE07u, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsICMP) {
  StringRef MIRString = R"(
   %cst0:_(s32) = G_CONSTANT i32 0
   %cst1:_(s32) = G_CONSTANT i32 1
   %icmp:_(s32) = G_ICMP intpred(ne), %cst0, %cst1
   %copy_icmp:_(s32) = COPY %icmp
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  // For targets that use 0 or 1 as icmp result in large register set high bits
  // to 0, does not analyze operands/compare predicate.
  EXPECT_EQ(0x00000000u, Res.One.getZExtValue());
  EXPECT_EQ(0xFFFFFFFEu, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsFCMP) {
  StringRef MIRString = R"(
   %cst0:_(s32) = G_FCONSTANT float 0.0
   %cst1:_(s32) = G_FCONSTANT float 1.0
   %fcmp:_(s32) = G_FCMP floatpred(one), %cst0, %cst1
   %copy_fcmp:_(s32) = COPY %fcmp
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  // For targets that use 0 or 1 as fcmp result in large register set high bits
  // to 0, does not analyze operands/compare predicate.
  EXPECT_EQ(0x00000000u, Res.One.getZExtValue());
  EXPECT_EQ(0xFFFFFFFEu, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsSelect) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 24
   %mask1:_(s8) = G_CONSTANT i8 224
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %mask2:_(s8) = G_CONSTANT i8 146
   %mask3:_(s8) = G_CONSTANT i8 36
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %cond:_(s1) = G_CONSTANT i1 false
   %select:_(s8) = G_SELECT %cond, %val0, %val1
   %copy_select:_(s8) = COPY %select
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(SrcReg);
  // Select KnownBits takes common bits of LHS and RHS, does not analyze
  // condition operand.
  //        111??000
  // select ?01?01?0
  //      = ??1????0
  EXPECT_EQ(0x20u, Res.One.getZExtValue());
  EXPECT_EQ(0x01u, Res.Zero.getZExtValue());
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIR);
  if (!TM)
    return;
  unsigned CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  unsigned SrcReg = FinalCopy->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Known = Info.getKnownBits(SrcReg);
  EXPECT_FALSE(Known.hasConflict());
  EXPECT_EQ(32u, Known.One.getZExtValue());
  EXPECT_EQ(95u, Known.Zero.getZExtValue());
  APInt Zeroes = Info.getKnownZeroes(SrcReg);
  EXPECT_EQ(Known.Zero, Zeroes);
}

TEST_F(AArch64GISelMITest, TestSignBitIsZero) {
  LLVMTargetMachine *TM = createTargetMachineAndModule();
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;
  Register CopySextLoad = Copies[Copies.size() - 2];
  Register CopySextNeg1 = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  EXPECT_EQ(25u, Info.computeNumSignBits(CopySextLoad));
  EXPECT_EQ(32u, Info.computeNumSignBits(CopySextNeg1));
}

TEST_F(AArch64GISelMITest, TestNumSignBitsSextInReg) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %load4:_(s32) = G_LOAD %ptr :: (load 4)

   %inreg7:_(s32) = G_SEXT_INREG %load4, 7
   %copy_inreg7:_(s32) = COPY %inreg7

   %inreg8:_(s32) = G_SEXT_INREG %load4, 8
   %copy_inreg8:_(s32) = COPY %inreg8

   %inreg9:_(s32) = G_SEXT_INREG %load4, 9
   %copy_inreg9:_(s32) = COPY %inreg9

   %inreg31:_(s32) = G_SEXT_INREG %load4, 31
   %copy_inreg31:_(s32) = COPY %inreg31

   %load1:_(s8) = G_LOAD %ptr :: (load 1)
   %sext_load1:_(s32) = G_SEXT %load1

   %inreg6_sext:_(s32) = G_SEXT_INREG %sext_load1, 6
   %copy_inreg6_sext:_(s32) = COPY %inreg6_sext

   %inreg7_sext:_(s32) = G_SEXT_INREG %sext_load1, 7
   %copy_inreg7_sext:_(s32) = COPY %inreg7_sext

   %inreg8_sext:_(s32) = G_SEXT_INREG %sext_load1, 8
   %copy_inreg8_sext:_(s32) = COPY %inreg8_sext

   %inreg9_sext:_(s32) = G_SEXT_INREG %sext_load1, 9
   %copy_inreg9_sext:_(s32) = COPY %inreg9_sext

   %inreg31_sext:_(s32) = G_SEXT_INREG %sext_load1, 31
   %copy_inreg31_sext:_(s32) = COPY %inreg31_sext
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

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

TEST_F(AArch64GISelMITest, TestNumSignBitsAssertSext) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %load4:_(s32) = G_LOAD %ptr :: (load 4)

   %assert_sext1:_(s32) = G_ASSERT_SEXT %load4, 1
   %copy_assert_sext1:_(s32) = COPY %assert_sext1

   %assert_sext7:_(s32) = G_ASSERT_SEXT %load4, 7
   %copy_assert_sext7:_(s32) = COPY %assert_sext7

   %assert_sext8:_(s32) = G_ASSERT_SEXT %load4, 8
   %copy_assert_sext8:_(s32) = COPY %assert_sext8

   %assert_sext9:_(s32) = G_ASSERT_SEXT %load4, 9
   %copy_assert_sext9:_(s32) = COPY %assert_sext9

   %assert_sext31:_(s32) = G_ASSERT_SEXT %load4, 31
   %copy_assert_sext31:_(s32) = COPY %assert_sext31

   %load1:_(s8) = G_LOAD %ptr :: (load 1)
   %sext_load1:_(s32) = G_SEXT %load1

   %assert_sext6_sext:_(s32) = G_ASSERT_SEXT %sext_load1, 6
   %copy_assert_sext6_sext:_(s32) = COPY %assert_sext6_sext

   %assert_sext7_sext:_(s32) = G_ASSERT_SEXT %sext_load1, 7
   %copy_assert_sext7_sext:_(s32) = COPY %assert_sext7_sext

   %assert_sext8_sext:_(s32) = G_ASSERT_SEXT %sext_load1, 8
   %copy_assert_sext8_sext:_(s32) = COPY %assert_sext8_sext

   %assert_sext9_sext:_(s32) = G_ASSERT_SEXT %sext_load1, 9
   %copy_assert_sext9_sext:_(s32) = COPY %assert_sext9_sext

   %assert_sext31_sext:_(s32) = G_ASSERT_SEXT %sext_load1, 31
   %copy_assert_sext31_sext:_(s32) = COPY %assert_sext31_sext
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

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
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
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

TEST_F(AMDGPUGISelMITest, TestNumSignBitsTrunc) {
  StringRef MIRString =
    "  %3:_(<4 x s32>) = G_IMPLICIT_DEF\n"
    "  %4:_(s32) = G_IMPLICIT_DEF\n"
    "  %5:_(s32) = G_AMDGPU_BUFFER_LOAD_UBYTE %3, %4, %4, %4, 0, 0, 0 :: (load 1)\n"
    "  %6:_(s32) = COPY %5\n"

    "  %7:_(s32) = G_AMDGPU_BUFFER_LOAD_SBYTE %3, %4, %4, %4, 0, 0, 0 :: (load 1)\n"
    "  %8:_(s32) = COPY %7\n"

    "  %9:_(s32) = G_AMDGPU_BUFFER_LOAD_USHORT %3, %4, %4, %4, 0, 0, 0 :: (load 2)\n"
    "  %10:_(s32) = COPY %9\n"

    "  %11:_(s32) = G_AMDGPU_BUFFER_LOAD_SSHORT %3, %4, %4, %4, 0, 0, 0 :: (load 2)\n"
    "  %12:_(s32) = COPY %11\n";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyLoadUByte = Copies[Copies.size() - 4];
  Register CopyLoadSByte = Copies[Copies.size() - 3];
  Register CopyLoadUShort = Copies[Copies.size() - 2];
  Register CopyLoadSShort = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);

  EXPECT_EQ(24u, Info.computeNumSignBits(CopyLoadUByte));
  EXPECT_EQ(25u, Info.computeNumSignBits(CopyLoadSByte));
  EXPECT_EQ(16u, Info.computeNumSignBits(CopyLoadUShort));
  EXPECT_EQ(17u, Info.computeNumSignBits(CopyLoadSShort));
}

TEST_F(AMDGPUGISelMITest, TestTargetKnownAlign) {
  StringRef MIRString =
    "  %5:_(p4) = G_INTRINSIC intrinsic(@llvm.amdgcn.dispatch.ptr)\n"
    "  %6:_(p4) = COPY %5\n"
    "  %7:_(p4) = G_INTRINSIC intrinsic(@llvm.amdgcn.queue.ptr)\n"
    "  %8:_(p4) = COPY %7\n"
    "  %9:_(p4) = G_INTRINSIC intrinsic(@llvm.amdgcn.kernarg.segment.ptr)\n"
    "  %10:_(p4) = COPY %9\n"
    "  %11:_(p4) = G_INTRINSIC intrinsic(@llvm.amdgcn.implicitarg.ptr)\n"
    "  %12:_(p4) = COPY %11\n"
    "  %13:_(p4) = G_INTRINSIC intrinsic(@llvm.amdgcn.implicit.buffer.ptr)\n"
    "  %14:_(p4) = COPY %13\n";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyDispatchPtr = Copies[Copies.size() - 5];
  Register CopyQueuePtr = Copies[Copies.size() - 4];
  Register CopyKernargSegmentPtr = Copies[Copies.size() - 3];
  Register CopyImplicitArgPtr = Copies[Copies.size() - 2];
  Register CopyImplicitBufferPtr = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);

  EXPECT_EQ(Align(4), Info.computeKnownAlignment(CopyDispatchPtr));
  EXPECT_EQ(Align(4), Info.computeKnownAlignment(CopyQueuePtr));
  EXPECT_EQ(Align(4), Info.computeKnownAlignment(CopyKernargSegmentPtr));
  EXPECT_EQ(Align(4), Info.computeKnownAlignment(CopyImplicitArgPtr));
  EXPECT_EQ(Align(4), Info.computeKnownAlignment(CopyImplicitBufferPtr));
}

TEST_F(AMDGPUGISelMITest, TestIsKnownToBeAPowerOfTwo) {

  StringRef MIRString = R"MIR(
  %zero:_(s32) = G_CONSTANT i32 0
  %one:_(s32) = G_CONSTANT i32 1
  %two:_(s32) = G_CONSTANT i32 2
  %three:_(s32) = G_CONSTANT i32 3
  %five:_(s32) = G_CONSTANT i32 5
  %copy_zero:_(s32) = COPY %zero
  %copy_one:_(s32) = COPY %one
  %copy_two:_(s32) = COPY %two
  %copy_three:_(s32) = COPY %three

  %trunc_two:_(s1) = G_TRUNC %two
  %trunc_three:_(s1) = G_TRUNC %three
  %trunc_five:_(s1) = G_TRUNC %five

  %copy_trunc_two:_(s1) = COPY %trunc_two
  %copy_trunc_three:_(s1) = COPY %trunc_three
  %copy_trunc_five:_(s1) = COPY %trunc_five

  %ptr:_(p1) = G_IMPLICIT_DEF
  %shift_amt:_(s32) = G_LOAD %ptr :: (load 4, addrspace 1)

  %shl_1:_(s32) = G_SHL %one, %shift_amt
  %copy_shl_1:_(s32) = COPY %shl_1

  %shl_2:_(s32) = G_SHL %two, %shift_amt
  %copy_shl_2:_(s32) = COPY %shl_2

  %not_sign_mask:_(s32) = G_LOAD %ptr :: (load 4, addrspace 1)
  %sign_mask:_(s32) = G_CONSTANT i32 -2147483648

  %lshr_not_sign_mask:_(s32) = G_LSHR %not_sign_mask, %shift_amt
  %copy_lshr_not_sign_mask:_(s32) = COPY %lshr_not_sign_mask

  %lshr_sign_mask:_(s32) = G_LSHR %sign_mask, %shift_amt
  %copy_lshr_sign_mask:_(s32) = COPY %lshr_sign_mask

  %or_pow2:_(s32) = G_OR %zero, %two
  %copy_or_pow2:_(s32) = COPY %or_pow2

)MIR";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

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

  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyShl1, *MRI, &KB));
  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyShl2, *MRI, &KB));

  EXPECT_FALSE(isKnownToBeAPowerOfTwo(CopyLShrNotSignMask, *MRI, &KB));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyLShrSignMask, *MRI, &KB));
  EXPECT_TRUE(isKnownToBeAPowerOfTwo(CopyOrPow2, *MRI, &KB));
}

TEST_F(AArch64GISelMITest, TestMetadata) {
  StringRef MIRString = "  %imp:_(p0) = G_IMPLICIT_DEF\n"
                        "  %load:_(s8) = G_LOAD %imp(p0) :: (load 1)\n"
                        "  %ext:_(s32) = G_ZEXT %load(s8)\n"
                        "  %cst:_(s32) = G_CONSTANT i32 1\n"
                        "  %and:_(s32) = G_AND %ext, %cst\n"
                        "  %copy:_(s32) = COPY %and(s32)\n";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  // We need a load with a metadata range for this to break. Fudge the load in
  // the string and replace it with something we can work with.
  MachineInstr *And = MRI->getVRegDef(SrcReg);
  MachineInstr *Ext = MRI->getVRegDef(And->getOperand(1).getReg());
  MachineInstr *Load = MRI->getVRegDef(Ext->getOperand(1).getReg());
  IntegerType *Int8Ty = Type::getInt8Ty(Context);

  // Value must be in [0, 2)
  Metadata *LowAndHigh[] = {
      ConstantAsMetadata::get(ConstantInt::get(Int8Ty, 0)),
      ConstantAsMetadata::get(ConstantInt::get(Int8Ty, 2))};
  auto NewMDNode = MDNode::get(Context, LowAndHigh);
  const MachineMemOperand *OldMMO = *Load->memoperands_begin();
  MachineMemOperand NewMMO(OldMMO->getPointerInfo(), OldMMO->getFlags(),
                           OldMMO->getSizeInBits(), OldMMO->getAlign(),
                           OldMMO->getAAInfo(), NewMDNode);
  MachineIRBuilder MIB(*Load);
  MIB.buildLoad(Load->getOperand(0), Load->getOperand(1), NewMMO);
  Load->eraseFromParent();

  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(And->getOperand(1).getReg());

  // We don't know what the result of the load is, so we don't know any ones.
  EXPECT_TRUE(Res.One.isNullValue());

  // We know that the value is in [0, 2). So, we don't know if the first bit
  // is 0 or not. However, we do know that every other bit must be 0.
  APInt Mask(Res.getBitWidth(), 1);
  Mask.flipAllBits();
  EXPECT_EQ(Mask.getZExtValue(), Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsExt) {
  StringRef MIRString = "  %c1:_(s16) = G_CONSTANT i16 1\n"
                        "  %x:_(s16) = G_IMPLICIT_DEF\n"
                        "  %y:_(s16) = G_AND %x, %c1\n"
                        "  %anyext:_(s32) = G_ANYEXT %y(s16)\n"
                        "  %r1:_(s32) = COPY %anyext\n"
                        "  %zext:_(s32) = G_ZEXT %y(s16)\n"
                        "  %r2:_(s32) = COPY %zext\n"
                        "  %sext:_(s32) = G_SEXT %y(s16)\n"
                        "  %r3:_(s32) = COPY %sext\n";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;
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

TEST_F(AArch64GISelMITest, TestKnownBitsSextInReg) {
  StringRef MIRString = R"(
   ; 000...0001
   %one:_(s32) = G_CONSTANT i32 1

   ; 000...0010
   %two:_(s32) = G_CONSTANT i32 2

   ; 000...1010
   %ten:_(s32) = G_CONSTANT i32 10

   ; ???...????
   %w0:_(s32) = COPY $w0

   ; ???...?1?
   %or:_(s32) = G_OR %w0, %two

   ; All bits are known.
   %inreg1:_(s32) = G_SEXT_INREG %one, 1
   %copy_inreg1:_(s32) = COPY %inreg1

   ; All bits unknown
   %inreg2:_(s32) = G_SEXT_INREG %or, 1
   %copy_inreg2:_(s32) = COPY %inreg2

   ; Extending from the only (known) set bit
   ; 111...11?
   %inreg3:_(s32) = G_SEXT_INREG %or, 2
   %copy_inreg3:_(s32) = COPY %inreg3

   ; Extending from a known set bit, overwriting all of the high set bits.
   ; 111...1110
   %inreg4:_(s32) = G_SEXT_INREG %ten, 2
   %copy_inreg4:_(s32) = COPY %inreg4

)";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;
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

TEST_F(AArch64GISelMITest, TestKnownBitsAssertSext) {
  StringRef MIRString = R"(
   ; 000...0001
   %one:_(s32) = G_CONSTANT i32 1

   ; 000...0010
   %two:_(s32) = G_CONSTANT i32 2

   ; 000...1010
   %ten:_(s32) = G_CONSTANT i32 10

   ; ???...????
   %w0:_(s32) = COPY $w0

   ; ???...?1?
   %or:_(s32) = G_OR %w0, %two

   ; All bits are known.
   %assert_sext1:_(s32) = G_ASSERT_SEXT %one, 1
   %copy_assert_sext1:_(s32) = COPY %assert_sext1

   ; All bits unknown
   %assert_sext2:_(s32) = G_ASSERT_SEXT %or, 1
   %copy_assert_sext2:_(s32) = COPY %assert_sext2

   ; Extending from the only (known) set bit
   ; 111...11?
   %assert_sext3:_(s32) = G_ASSERT_SEXT %or, 2
   %copy_assert_sext3:_(s32) = COPY %assert_sext3

   ; Extending from a known set bit, overwriting all of the high set bits.
   ; 111...1110
   %assert_sext4:_(s32) = G_ASSERT_SEXT %ten, 2
   %copy_assert_sext4:_(s32) = COPY %assert_sext4
)";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;
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

TEST_F(AArch64GISelMITest, TestKnownBitsMergeValues) {
  StringRef MIRString = R"(
   %val0:_(s16) = G_CONSTANT i16 35224
   %val1:_(s16) = G_CONSTANT i16 17494
   %val2:_(s16) = G_CONSTANT i16 4659
   %val3:_(s16) = G_CONSTANT i16 43981
   %merge:_(s64) = G_MERGE_VALUES %val0, %val1, %val2, %val3
   %mergecopy:_(s64) = COPY %merge
)";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  const uint64_t TestVal = UINT64_C(0xabcd123344568998);
  Register CopyMerge = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);
  KnownBits Res = Info.getKnownBits(CopyMerge);
  EXPECT_EQ(64u, Res.getBitWidth());
  EXPECT_EQ(TestVal, Res.One.getZExtValue());
  EXPECT_EQ(~TestVal, Res.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsUnmergeValues) {
  StringRef MIRString = R"(
   %val:_(s64) = G_CONSTANT i64 12379570962110515608
   %val0:_(s16), %val1:_(s16), %val2:_(s16), %val3:_(s16) = G_UNMERGE_VALUES %val
   %part0:_(s16) = COPY %val0
   %part1:_(s16) = COPY %val1
   %part2:_(s16) = COPY %val2
   %part3:_(s16) = COPY %val3

)";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  const uint64_t TestVal = UINT64_C(0xabcd123344568998);
  GISelKnownBits Info(*MF);

  int Offset = -4;
  for (unsigned BitOffset = 0; BitOffset != 64; BitOffset += 16, ++Offset) {
    Register Part = Copies[Copies.size() + Offset];
    KnownBits PartKnown = Info.getKnownBits(Part);
    EXPECT_EQ(16u, PartKnown.getBitWidth());

    uint16_t PartTestVal = static_cast<uint16_t>(TestVal >> BitOffset);
    EXPECT_EQ(PartTestVal, PartKnown.One.getZExtValue());
    EXPECT_EQ(static_cast<uint16_t>(~PartTestVal), PartKnown.Zero.getZExtValue());
  }
}

TEST_F(AArch64GISelMITest, TestKnownBitsBSwapBitReverse) {
  StringRef MIRString = R"(
   %const:_(s32) = G_CONSTANT i32 287454020
   %bswap:_(s32) = G_BSWAP %const
   %bitreverse:_(s32) = G_BITREVERSE %const
   %copy_bswap:_(s32) = COPY %bswap
   %copy_bitreverse:_(s32) = COPY %bitreverse
)";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  const uint32_t TestVal = 0x11223344;

  Register CopyBSwap = Copies[Copies.size() - 2];
  Register CopyBitReverse = Copies[Copies.size() - 1];

  GISelKnownBits Info(*MF);

  KnownBits BSwapKnown = Info.getKnownBits(CopyBSwap);
  EXPECT_EQ(32u, BSwapKnown.getBitWidth());
  EXPECT_EQ(TestVal, BSwapKnown.One.getZExtValue());
  EXPECT_EQ(~TestVal, BSwapKnown.Zero.getZExtValue());

  KnownBits BitReverseKnown = Info.getKnownBits(CopyBitReverse);
  EXPECT_EQ(32u, BitReverseKnown.getBitWidth());
  EXPECT_EQ(TestVal, BitReverseKnown.One.getZExtValue());
  EXPECT_EQ(~TestVal, BitReverseKnown.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsUMAX) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 10
   %mask1:_(s8) = G_CONSTANT i8 1
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %mask2:_(s8) = G_CONSTANT i8 3
   %mask3:_(s8) = G_CONSTANT i8 12
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %umax0:_(s8) = G_UMAX %val0, %val1
   %copy_umax0:_(s8) = COPY %umax0

   %mask4:_(s8) = G_CONSTANT i8 14
   %mask5:_(s8) = G_CONSTANT i8 2
   %tmp3:_(s8) = G_AND %unknown, %mask4
   %val3:_(s8) = G_OR %tmp3, %mask5
   %mask6:_(s8) = G_CONSTANT i8 4
   %mask7:_(s8) = G_CONSTANT i8 11
   %tmp4:_(s8) = G_AND %unknown, %mask6
   %val4:_(s8) = G_OR %tmp4, %mask7
   %umax1:_(s8) = G_UMAX %val3, %val4
   %copy_umax1:_(s8) = COPY %umax1
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg0 = Copies[Copies.size() - 2];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  // Compares min/max of LHS and RHS, min uses 0 for unknown bits, max uses 1.
  // If min(LHS) >= max(RHS) returns KnownBits for LHS, similar for RHS. If this
  // fails tries to calculate individual bits: common bits for both operands and
  // a few leading bits in some cases.
  //      0000?0?1
  // umax 000011??
  //    = 000011??
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  EXPECT_EQ(0x0Cu, Res0.One.getZExtValue());
  EXPECT_EQ(0xF0u, Res0.Zero.getZExtValue());

  Register CopyReg1 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy1 = MRI->getVRegDef(CopyReg1);
  Register SrcReg1 = FinalCopy1->getOperand(1).getReg();
  KnownBits Res1 = Info.getKnownBits(SrcReg1);
  //      0000??10
  // umax 00001?11
  //    = 00001?1?
  EXPECT_EQ(0x0Au, Res1.One.getZExtValue());
  EXPECT_EQ(0xF0u, Res1.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsUMax) {
  StringRef MIRString = R"(
   %val:_(s32) = COPY $w0
   %zext:_(s64) = G_ZEXT %val
   %const:_(s64) = G_CONSTANT i64 -256
   %umax:_(s64) = G_UMAX %zext, %const
   %copy_umax:_(s64) = COPY %umax
)";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyUMax = Copies[Copies.size() - 1];
  GISelKnownBits Info(*MF);

  KnownBits KnownUmax = Info.getKnownBits(CopyUMax);
  EXPECT_EQ(64u, KnownUmax.getBitWidth());
  EXPECT_EQ(0xffu, KnownUmax.Zero.getZExtValue());
  EXPECT_EQ(0xffffffffffffff00, KnownUmax.One.getZExtValue());

  EXPECT_EQ(0xffu, KnownUmax.Zero.getZExtValue());
  EXPECT_EQ(0xffffffffffffff00, KnownUmax.One.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsUMIN) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 10
   %mask1:_(s8) = G_CONSTANT i8 1
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %mask2:_(s8) = G_CONSTANT i8 3
   %mask3:_(s8) = G_CONSTANT i8 12
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %umin:_(s8) = G_UMIN %val0, %val1
   %copy_umin:_(s8) = COPY %umin
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg0 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  // Flips the range of operands: [0, 0xFFFFFFFF] <-> [0xFFFFFFFF, 0],
  // uses umax and flips result back.
  //      0000?0?1
  // umin 000011??
  //    = 0000?0?1
  EXPECT_EQ(0x01u, Res0.One.getZExtValue());
  EXPECT_EQ(0xF4u, Res0.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsSMAX) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 128
   %mask1:_(s8) = G_CONSTANT i8 64
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %mask2:_(s8) = G_CONSTANT i8 1
   %mask3:_(s8) = G_CONSTANT i8 128
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %smax:_(s8) = G_SMAX %val0, %val1
   %copy_smax:_(s8) = COPY %smax
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg0 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  // Flips the range of operands: [-0x80000000, 0x7FFFFFFF] <-> [0, 0xFFFFFFFF],
  // uses umax and flips result back.
  // RHS is negative, LHS is either positive or negative with smaller abs value.
  //      ?1000000
  // smax 1000000?
  //    = ?1000000
  EXPECT_EQ(0x40u, Res0.One.getZExtValue());
  EXPECT_EQ(0x3Fu, Res0.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsSMIN) {
  StringRef MIRString = R"(
   %ptr:_(p0) = G_IMPLICIT_DEF
   %unknown:_(s8) = G_LOAD %ptr(p0) :: (load 1)
   %mask0:_(s8) = G_CONSTANT i8 128
   %mask1:_(s8) = G_CONSTANT i8 64
   %tmp0:_(s8) = G_AND %unknown, %mask0
   %val0:_(s8) = G_OR %tmp0, %mask1
   %mask2:_(s8) = G_CONSTANT i8 1
   %mask3:_(s8) = G_CONSTANT i8 128
   %tmp1:_(s8) = G_AND %unknown, %mask2
   %val1:_(s8) = G_OR %tmp1, %mask3
   %smin:_(s8) = G_SMIN %val0, %val1
   %copy_smin:_(s8) = COPY %smin
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register CopyReg0 = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy0 = MRI->getVRegDef(CopyReg0);
  Register SrcReg0 = FinalCopy0->getOperand(1).getReg();
  GISelKnownBits Info(*MF);
  KnownBits Res0 = Info.getKnownBits(SrcReg0);
  // Flips the range of operands: [-0x80000000, 0x7FFFFFFF] <-> [0xFFFFFFFF, 0],
  // uses umax and flips result back.
  // RHS is negative, LHS is either positive or negative with smaller abs value.
  //      ?1000000
  // smin 1000000?
  //    = 1000000?
  EXPECT_EQ(0x80u, Res0.One.getZExtValue());
  EXPECT_EQ(0x7Eu, Res0.Zero.getZExtValue());
}

TEST_F(AArch64GISelMITest, TestInvalidQueries) {
  StringRef MIRString = R"(
   %src:_(s32) = COPY $w0
   %thirty2:_(s32) = G_CONSTANT i32 32
   %equalSized:_(s32) = G_SHL %src, %thirty2
   %copy1:_(s32) = COPY %equalSized
   %thirty3:_(s32) = G_CONSTANT i32 33
   %biggerSized:_(s32) = G_SHL %src, %thirty3
   %copy2:_(s32) = COPY %biggerSized
)";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

  Register EqSizedCopyReg = Copies[Copies.size() - 2];
  MachineInstr *EqSizedCopy = MRI->getVRegDef(EqSizedCopyReg);
  Register EqSizedShl = EqSizedCopy->getOperand(1).getReg();

  Register BiggerSizedCopyReg = Copies[Copies.size() - 1];
  MachineInstr *BiggerSizedCopy = MRI->getVRegDef(BiggerSizedCopyReg);
  Register BiggerSizedShl = BiggerSizedCopy->getOperand(1).getReg();

  GISelKnownBits Info(*MF);
  KnownBits EqSizeRes = Info.getKnownBits(EqSizedShl);
  KnownBits BiggerSizeRes = Info.getKnownBits(BiggerSizedShl);


  // We don't know what the result of the shift is, but we should not crash
  EXPECT_TRUE(EqSizeRes.One.isNullValue());
  EXPECT_TRUE(EqSizeRes.Zero.isNullValue());

  EXPECT_TRUE(BiggerSizeRes.One.isNullValue());
  EXPECT_TRUE(BiggerSizeRes.Zero.isNullValue());
}

TEST_F(AArch64GISelMITest, TestKnownBitsAssertZext) {
  StringRef MIRString = R"(
   %copy:_(s64) = COPY $x0

   %assert8:_(s64) = G_ASSERT_ZEXT %copy, 8
   %copy_assert8:_(s64) = COPY %assert8

   %assert1:_(s64) = G_ASSERT_ZEXT %copy, 1
   %copy_assert1:_(s64) = COPY %assert1

   %assert63:_(s64) = G_ASSERT_ZEXT %copy, 63
   %copy_assert63:_(s64) = COPY %assert63

   %assert3:_(s64) = G_ASSERT_ZEXT %copy, 3
   %copy_assert3:_(s64) = COPY %assert3
)";

  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString);
  if (!TM)
    return;

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
