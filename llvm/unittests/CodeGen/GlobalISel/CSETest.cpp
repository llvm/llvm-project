//===- CSETest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"

namespace {

TEST_F(AArch64GISelMITest, TestCSE) {
  setUp();
  if (!TM)
    return;

  LLT s16{LLT::scalar(16)};
  LLT s32{LLT::scalar(32)};
  auto MIBInput = B.buildInstr(TargetOpcode::G_TRUNC, {s16}, {Copies[0]});
  auto MIBInput1 = B.buildInstr(TargetOpcode::G_TRUNC, {s16}, {Copies[1]});
  auto MIBAdd = B.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  GISelCSEInfo CSEInfo;
  CSEInfo.setCSEConfig(std::make_unique<CSEConfigFull>());
  CSEInfo.analyze(*MF);
  B.setCSEInfo(&CSEInfo);
  CSEMIRBuilder CSEB(B.getState());

  CSEB.setInsertPt(*EntryMBB, EntryMBB->begin());
  unsigned AddReg = MRI->createGenericVirtualRegister(s16);
  auto MIBAddCopy =
      CSEB.buildInstr(TargetOpcode::G_ADD, {AddReg}, {MIBInput, MIBInput});
  EXPECT_EQ(MIBAddCopy->getOpcode(), TargetOpcode::COPY);
  auto MIBAdd2 =
      CSEB.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  EXPECT_TRUE(&*MIBAdd == &*MIBAdd2);
  auto MIBAdd4 =
      CSEB.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  EXPECT_TRUE(&*MIBAdd == &*MIBAdd4);
  auto MIBAdd5 =
      CSEB.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput1});
  EXPECT_TRUE(&*MIBAdd != &*MIBAdd5);

  // Try building G_CONSTANTS.
  auto MIBCst = CSEB.buildConstant(s32, 0);
  auto MIBCst1 = CSEB.buildConstant(s32, 0);
  EXPECT_TRUE(&*MIBCst == &*MIBCst1);
  // Try the CFing of BinaryOps.
  auto MIBCF1 = CSEB.buildInstr(TargetOpcode::G_ADD, {s32}, {MIBCst, MIBCst});
  EXPECT_TRUE(&*MIBCF1 == &*MIBCst);

  // Try out building FCONSTANTs.
  auto MIBFP0 = CSEB.buildFConstant(s32, 1.0);
  auto MIBFP0_1 = CSEB.buildFConstant(s32, 1.0);
  EXPECT_TRUE(&*MIBFP0 == &*MIBFP0_1);
  CSEInfo.print();

  // Make sure buildConstant with a vector type doesn't crash, and the elements
  // CSE.
  auto Splat0 = CSEB.buildConstant(LLT::vector(2, s32), 0);
  EXPECT_EQ(TargetOpcode::G_BUILD_VECTOR, Splat0->getOpcode());
  EXPECT_EQ(Splat0.getReg(1), Splat0.getReg(2));
  EXPECT_EQ(&*MIBCst, MRI->getVRegDef(Splat0.getReg(1)));

  auto FSplat = CSEB.buildFConstant(LLT::vector(2, s32), 1.0);
  EXPECT_EQ(TargetOpcode::G_BUILD_VECTOR, FSplat->getOpcode());
  EXPECT_EQ(FSplat.getReg(1), FSplat.getReg(2));
  EXPECT_EQ(&*MIBFP0, MRI->getVRegDef(FSplat.getReg(1)));

  // Check G_UNMERGE_VALUES
  auto MIBUnmerge = CSEB.buildUnmerge({s32, s32}, Copies[0]);
  auto MIBUnmerge2 = CSEB.buildUnmerge({s32, s32}, Copies[0]);
  EXPECT_TRUE(&*MIBUnmerge == &*MIBUnmerge2);

  // Check G_IMPLICIT_DEF
  auto Undef0 = CSEB.buildUndef(s32);
  auto Undef1 = CSEB.buildUndef(s32);
  EXPECT_EQ(&*Undef0, &*Undef1);

  // If the observer is installed to the MF, CSE can also
  // track new instructions built without the CSEBuilder and
  // the newly built instructions are available for CSEing next
  // time a build call is made through the CSEMIRBuilder.
  // Additionally, the CSE implementation lazily hashes instructions
  // (every build call) to give chance for the instruction to be fully
  // built (say using .addUse().addDef().. so on).
  GISelObserverWrapper WrapperObserver(&CSEInfo);
  RAIIMFObsDelInstaller Installer(*MF, WrapperObserver);
  MachineIRBuilder RegularBuilder(*MF);
  RegularBuilder.setInsertPt(*EntryMBB, EntryMBB->begin());
  auto NonCSEFMul = RegularBuilder.buildInstr(TargetOpcode::G_AND)
                        .addDef(MRI->createGenericVirtualRegister(s32))
                        .addUse(Copies[0])
                        .addUse(Copies[1]);
  auto CSEFMul =
      CSEB.buildInstr(TargetOpcode::G_AND, {s32}, {Copies[0], Copies[1]});
  EXPECT_EQ(&*CSEFMul, &*NonCSEFMul);

  auto ExtractMIB = CSEB.buildInstr(TargetOpcode::G_EXTRACT, {s16},
                                    {Copies[0], static_cast<uint64_t>(0)});
  auto ExtractMIB1 = CSEB.buildInstr(TargetOpcode::G_EXTRACT, {s16},
                                     {Copies[0], static_cast<uint64_t>(0)});
  auto ExtractMIB2 = CSEB.buildInstr(TargetOpcode::G_EXTRACT, {s16},
                                     {Copies[0], static_cast<uint64_t>(1)});
  EXPECT_EQ(&*ExtractMIB, &*ExtractMIB1);
  EXPECT_NE(&*ExtractMIB, &*ExtractMIB2);
}

TEST_F(AArch64GISelMITest, TestCSEConstantConfig) {
  setUp();
  if (!TM)
    return;

  LLT s16{LLT::scalar(16)};
  auto MIBInput = B.buildInstr(TargetOpcode::G_TRUNC, {s16}, {Copies[0]});
  auto MIBAdd = B.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  auto MIBZero = B.buildConstant(s16, 0);
  GISelCSEInfo CSEInfo;
  CSEInfo.setCSEConfig(std::make_unique<CSEConfigConstantOnly>());
  CSEInfo.analyze(*MF);
  B.setCSEInfo(&CSEInfo);
  CSEMIRBuilder CSEB(B.getState());
  CSEB.setInsertPt(*EntryMBB, EntryMBB->begin());
  auto MIBAdd1 =
      CSEB.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  // We should CSE constants only. Adds should not be CSEd.
  EXPECT_TRUE(MIBAdd1->getOpcode() != TargetOpcode::COPY);
  EXPECT_TRUE(&*MIBAdd1 != &*MIBAdd);
  // We should CSE constant.
  auto MIBZeroTmp = CSEB.buildConstant(s16, 0);
  EXPECT_TRUE(&*MIBZero == &*MIBZeroTmp);

  // Check G_IMPLICIT_DEF
  auto Undef0 = CSEB.buildUndef(s16);
  auto Undef1 = CSEB.buildUndef(s16);
  EXPECT_EQ(&*Undef0, &*Undef1);
}
} // namespace
