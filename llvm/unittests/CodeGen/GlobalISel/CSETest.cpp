//===- CSETest.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"

namespace {

TEST_F(GISelMITest, TestCSE) {
  if (!TM)
    return;

  LLT s16{LLT::scalar(16)};
  LLT s32{LLT::scalar(32)};
  auto MIBInput = B.buildInstr(TargetOpcode::G_TRUNC, {s16}, {Copies[0]});
  auto MIBInput1 = B.buildInstr(TargetOpcode::G_TRUNC, {s16}, {Copies[1]});
  auto MIBAdd = B.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  GISelCSEInfo CSEInfo;
  CSEInfo.setCSEConfig(make_unique<CSEConfig>());
  CSEInfo.analyze(*MF);
  B.setCSEInfo(&CSEInfo);
  CSEMIRBuilder CSEB(B.getState());
  CSEB.setInsertPt(*EntryMBB, EntryMBB->begin());
  unsigned AddReg = MRI->createGenericVirtualRegister(s16);
  auto MIBAddCopy =
      CSEB.buildInstr(TargetOpcode::G_ADD, {AddReg}, {MIBInput, MIBInput});
  ASSERT_EQ(MIBAddCopy->getOpcode(), TargetOpcode::COPY);
  auto MIBAdd2 =
      CSEB.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  ASSERT_TRUE(&*MIBAdd == &*MIBAdd2);
  auto MIBAdd4 =
      CSEB.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  ASSERT_TRUE(&*MIBAdd == &*MIBAdd4);
  auto MIBAdd5 =
      CSEB.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput1});
  ASSERT_TRUE(&*MIBAdd != &*MIBAdd5);

  // Try building G_CONSTANTS.
  auto MIBCst = CSEB.buildConstant(s32, 0);
  auto MIBCst1 = CSEB.buildConstant(s32, 0);
  ASSERT_TRUE(&*MIBCst == &*MIBCst1);
  // Try the CFing of BinaryOps.
  auto MIBCF1 = CSEB.buildInstr(TargetOpcode::G_ADD, {s32}, {MIBCst, MIBCst});
  ASSERT_TRUE(&*MIBCF1 == &*MIBCst);

  // Try out building FCONSTANTs.
  auto MIBFP0 = CSEB.buildFConstant(s32, 1.0);
  auto MIBFP0_1 = CSEB.buildFConstant(s32, 1.0);
  ASSERT_TRUE(&*MIBFP0 == &*MIBFP0_1);
  CSEInfo.print();

  // Check G_UNMERGE_VALUES
  auto MIBUnmerge = CSEB.buildUnmerge({s32, s32}, Copies[0]);
  auto MIBUnmerge2 = CSEB.buildUnmerge({s32, s32}, Copies[0]);
  ASSERT_TRUE(&*MIBUnmerge == &*MIBUnmerge2);
}

TEST_F(GISelMITest, TestCSEConstantConfig) {
  if (!TM)
    return;

  LLT s16{LLT::scalar(16)};
  auto MIBInput = B.buildInstr(TargetOpcode::G_TRUNC, {s16}, {Copies[0]});
  auto MIBAdd = B.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  auto MIBZero = B.buildConstant(s16, 0);
  GISelCSEInfo CSEInfo;
  CSEInfo.setCSEConfig(make_unique<CSEConfigConstantOnly>());
  CSEInfo.analyze(*MF);
  B.setCSEInfo(&CSEInfo);
  CSEMIRBuilder CSEB(B.getState());
  CSEB.setInsertPt(*EntryMBB, EntryMBB->begin());
  auto MIBAdd1 =
      CSEB.buildInstr(TargetOpcode::G_ADD, {s16}, {MIBInput, MIBInput});
  // We should CSE constants only. Adds should not be CSEd.
  ASSERT_TRUE(MIBAdd1->getOpcode() != TargetOpcode::COPY);
  ASSERT_TRUE(&*MIBAdd1 != &*MIBAdd);
  // We should CSE constant.
  auto MIBZeroTmp = CSEB.buildConstant(s16, 0);
  ASSERT_TRUE(&*MIBZero == &*MIBZeroTmp);
}
} // namespace
