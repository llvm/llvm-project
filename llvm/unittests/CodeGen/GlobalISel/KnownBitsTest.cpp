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

TEST_F(GISelMITest, TestKnownBitsCst) {
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

TEST_F(GISelMITest, TestKnownBitsCstWithClass) {
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

TEST_F(GISelMITest, TestKnownBitsPtrToIntViceVersa) {
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
TEST_F(GISelMITest, TestKnownBitsXOR) {
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

TEST_F(GISelMITest, TestKnownBits) {

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

TEST_F(GISelMITest, TestSignBitIsZero) {
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
