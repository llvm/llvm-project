//===- CSETest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"
#include "gtest/gtest.h"

namespace {

TEST_F(AArch64GISelMITest, TestCSE) {
  setUp();
  if (!TM)
    GTEST_SKIP();

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

  CSEB.setInsertPt(B.getMBB(), B.getInsertPt());
  Register AddReg = MRI->createGenericVirtualRegister(s16);
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
  auto Splat0 = CSEB.buildConstant(LLT::fixed_vector(2, s32), 0);
  EXPECT_EQ(TargetOpcode::G_BUILD_VECTOR, Splat0->getOpcode());
  EXPECT_EQ(Splat0.getReg(1), Splat0.getReg(2));
  EXPECT_EQ(&*MIBCst, MRI->getVRegDef(Splat0.getReg(1)));

  auto FSplat = CSEB.buildFConstant(LLT::fixed_vector(2, s32), 1.0);
  EXPECT_EQ(TargetOpcode::G_BUILD_VECTOR, FSplat->getOpcode());
  EXPECT_EQ(FSplat.getReg(1), FSplat.getReg(2));
  EXPECT_EQ(&*MIBFP0, MRI->getVRegDef(FSplat.getReg(1)));

  // Check G_UNMERGE_VALUES
  auto MIBUnmerge = CSEB.buildUnmerge({s32, s32}, Copies[0]);
  auto MIBUnmerge2 = CSEB.buildUnmerge({s32, s32}, Copies[0]);
  EXPECT_TRUE(&*MIBUnmerge == &*MIBUnmerge2);

  // Check G_BUILD_VECTOR
  Register Reg1 = MRI->createGenericVirtualRegister(s32);
  Register Reg2 = MRI->createGenericVirtualRegister(s32);
  auto BuildVec1 =
      CSEB.buildBuildVector(LLT::fixed_vector(4, 32), {Reg1, Reg2, Reg1, Reg2});
  auto BuildVec2 =
      CSEB.buildBuildVector(LLT::fixed_vector(4, 32), {Reg1, Reg2, Reg1, Reg2});
  EXPECT_EQ(TargetOpcode::G_BUILD_VECTOR, BuildVec1->getOpcode());
  EXPECT_EQ(TargetOpcode::G_BUILD_VECTOR, BuildVec2->getOpcode());
  EXPECT_TRUE(&*BuildVec1 == &*BuildVec2);

  // Check G_BUILD_VECTOR_TRUNC
  auto BuildVecTrunc1 = CSEB.buildBuildVectorTrunc(LLT::fixed_vector(4, 16),
                                                   {Reg1, Reg2, Reg1, Reg2});
  auto BuildVecTrunc2 = CSEB.buildBuildVectorTrunc(LLT::fixed_vector(4, 16),
                                                   {Reg1, Reg2, Reg1, Reg2});
  EXPECT_EQ(TargetOpcode::G_BUILD_VECTOR_TRUNC, BuildVecTrunc1->getOpcode());
  EXPECT_EQ(TargetOpcode::G_BUILD_VECTOR_TRUNC, BuildVecTrunc2->getOpcode());
  EXPECT_TRUE(&*BuildVecTrunc1 == &*BuildVecTrunc2);

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


  auto SextInRegMIB = CSEB.buildSExtInReg(s16, Copies[0], 0);
  auto SextInRegMIB1 = CSEB.buildSExtInReg(s16, Copies[0], 0);
  auto SextInRegMIB2 = CSEB.buildSExtInReg(s16, Copies[0], 1);
  EXPECT_EQ(&*SextInRegMIB, &*SextInRegMIB1);
  EXPECT_NE(&*SextInRegMIB, &*SextInRegMIB2);
}

TEST_F(AArch64GISelMITest, TestCSEConstantConfig) {
  setUp();
  if (!TM)
    GTEST_SKIP();

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

TEST_F(AArch64GISelMITest, TestCSEImmediateNextCSE) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  LLT s32{LLT::scalar(32)};
  // We want to check that when the CSE hit is on the next instruction, i.e. at
  // the current insert pt, that the insertion point is moved ahead of the
  // instruction.

  GISelCSEInfo CSEInfo;
  CSEInfo.setCSEConfig(std::make_unique<CSEConfigConstantOnly>());
  CSEInfo.analyze(*MF);
  B.setCSEInfo(&CSEInfo);
  CSEMIRBuilder CSEB(B.getState());
  CSEB.buildConstant(s32, 0);
  auto MIBCst2 = CSEB.buildConstant(s32, 2);

  // Move the insert point before the second constant.
  CSEB.setInsertPt(CSEB.getMBB(), --CSEB.getInsertPt());
  auto MIBCst3 = CSEB.buildConstant(s32, 2);
  EXPECT_TRUE(&*MIBCst2 == &*MIBCst3);
  EXPECT_TRUE(CSEB.getInsertPt() == CSEB.getMBB().end());
}

TEST_F(AArch64GISelMITest, TestConstantFoldCTL) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  LLT s32 = LLT::scalar(32);

  GISelCSEInfo CSEInfo;
  CSEInfo.setCSEConfig(std::make_unique<CSEConfigConstantOnly>());
  CSEInfo.analyze(*MF);
  B.setCSEInfo(&CSEInfo);
  CSEMIRBuilder CSEB(B.getState());
  auto Cst8 = CSEB.buildConstant(s32, 8);
  auto *CtlzDef = &*CSEB.buildCTLZ(s32, Cst8);
  EXPECT_TRUE(CtlzDef->getOpcode() == TargetOpcode::G_CONSTANT);
  EXPECT_TRUE(CtlzDef->getOperand(1).getCImm()->getZExtValue() == 28);

  // Test vector.
  auto Cst16 = CSEB.buildConstant(s32, 16);
  auto Cst32 = CSEB.buildConstant(s32, 32);
  auto Cst64 = CSEB.buildConstant(s32, 64);
  LLT VecTy = LLT::fixed_vector(4, s32);
  auto BV = CSEB.buildBuildVector(VecTy, {Cst8.getReg(0), Cst16.getReg(0),
                                          Cst32.getReg(0), Cst64.getReg(0)});
  CSEB.buildCTLZ(VecTy, BV);

  auto CheckStr = R"(
  ; CHECK: [[CST8:%[0-9]+]]:_(s32) = G_CONSTANT i32 8
  ; CHECK: [[CST28:%[0-9]+]]:_(s32) = G_CONSTANT i32 28
  ; CHECK: [[CST16:%[0-9]+]]:_(s32) = G_CONSTANT i32 16
  ; CHECK: [[CST32:%[0-9]+]]:_(s32) = G_CONSTANT i32 32
  ; CHECK: [[CST64:%[0-9]+]]:_(s32) = G_CONSTANT i32 64
  ; CHECK: [[BV1:%[0-9]+]]:_(<4 x s32>) = G_BUILD_VECTOR [[CST8]]:_(s32), [[CST16]]:_(s32), [[CST32]]:_(s32), [[CST64]]:_(s32)
  ; CHECK: [[CST27:%[0-9]+]]:_(s32) = G_CONSTANT i32 27
  ; CHECK: [[CST26:%[0-9]+]]:_(s32) = G_CONSTANT i32 26
  ; CHECK: [[CST25:%[0-9]+]]:_(s32) = G_CONSTANT i32 25
  ; CHECK: [[BV2:%[0-9]+]]:_(<4 x s32>) = G_BUILD_VECTOR [[CST28]]:_(s32), [[CST27]]:_(s32), [[CST26]]:_(s32), [[CST25]]:_(s32)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

} // namespace
