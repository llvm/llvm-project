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

TEST_F(AArch64GISelMITest, TestConstantFoldCTT) {
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
  auto *CttzDef = &*CSEB.buildCTTZ(s32, Cst8);
  EXPECT_TRUE(CttzDef->getOpcode() == TargetOpcode::G_CONSTANT);
  EXPECT_TRUE(CttzDef->getOperand(1).getCImm()->getZExtValue() == 3);

  // Test vector.
  auto Cst16 = CSEB.buildConstant(s32, 16);
  auto Cst32 = CSEB.buildConstant(s32, 32);
  auto Cst64 = CSEB.buildConstant(s32, 64);
  LLT VecTy = LLT::fixed_vector(4, s32);
  auto BV = CSEB.buildBuildVector(VecTy, {Cst8.getReg(0), Cst16.getReg(0),
                                          Cst32.getReg(0), Cst64.getReg(0)});
  CSEB.buildCTTZ(VecTy, BV);

  auto CheckStr = R"(
  ; CHECK: [[CST8:%[0-9]+]]:_(s32) = G_CONSTANT i32 8
  ; CHECK: [[CST3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
  ; CHECK: [[CST16:%[0-9]+]]:_(s32) = G_CONSTANT i32 16
  ; CHECK: [[CST32:%[0-9]+]]:_(s32) = G_CONSTANT i32 32
  ; CHECK: [[CST64:%[0-9]+]]:_(s32) = G_CONSTANT i32 64
  ; CHECK: [[BV1:%[0-9]+]]:_(<4 x s32>) = G_BUILD_VECTOR [[CST8]]:_(s32), [[CST16]]:_(s32), [[CST32]]:_(s32), [[CST64]]:_(s32)
  ; CHECK: [[CST27:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
  ; CHECK: [[CST26:%[0-9]+]]:_(s32) = G_CONSTANT i32 5
  ; CHECK: [[CST25:%[0-9]+]]:_(s32) = G_CONSTANT i32 6
  ; CHECK: [[BV2:%[0-9]+]]:_(<4 x s32>) = G_BUILD_VECTOR [[CST3]]:_(s32), [[CST27]]:_(s32), [[CST26]]:_(s32), [[CST25]]:_(s32)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(AArch64GISelMITest, TestConstantFoldICMP) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  LLT s32 = LLT::scalar(32);
  LLT s1 = LLT::scalar(1);

  GISelCSEInfo CSEInfo;
  CSEInfo.setCSEConfig(std::make_unique<CSEConfigConstantOnly>());
  CSEInfo.analyze(*MF);
  B.setCSEInfo(&CSEInfo);
  CSEMIRBuilder CSEB(B.getState());

  auto One = CSEB.buildConstant(s32, 1);
  auto Two = CSEB.buildConstant(s32, 2);
  auto MinusOne = CSEB.buildConstant(s32, -1);
  auto MinusTwo = CSEB.buildConstant(s32, -2);

  // ICMP_EQ
  {
    auto I = CSEB.buildICmp(CmpInst::Predicate::ICMP_EQ, s1, One, One);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  // ICMP_NE
  {
    auto I = CSEB.buildICmp(CmpInst::Predicate::ICMP_NE, s1, One, Two);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  // ICMP_UGT
  {
    auto I = CSEB.buildICmp(CmpInst::Predicate::ICMP_UGT, s1, Two, One);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  // ICMP_UGE
  {
    auto I = CSEB.buildICmp(CmpInst::Predicate::ICMP_UGE, s1, One, One);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  // ICMP_ULT
  {
    auto I = CSEB.buildICmp(CmpInst::Predicate::ICMP_ULT, s1, One, Two);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  // ICMP_ULE
  {
    auto I = CSEB.buildICmp(CmpInst::Predicate::ICMP_ULE, s1, Two, Two);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  // ICMP_SGT
  {
    auto I =
        CSEB.buildICmp(CmpInst::Predicate::ICMP_SGT, s1, MinusOne, MinusTwo);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  // ICMP_SGE
  {
    auto I =
        CSEB.buildICmp(CmpInst::Predicate::ICMP_SGE, s1, MinusOne, MinusOne);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  // ICMP_SLT
  {
    auto I =
        CSEB.buildICmp(CmpInst::Predicate::ICMP_SLT, s1, MinusTwo, MinusOne);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  // ICMP_SLE
  {
    auto I =
        CSEB.buildICmp(CmpInst::Predicate::ICMP_SLE, s1, MinusTwo, MinusOne);
    EXPECT_TRUE(I->getOpcode() == TargetOpcode::G_CONSTANT);
    EXPECT_TRUE(I->getOperand(1).getCImm()->getZExtValue());
  }

  LLT VecTy = LLT::fixed_vector(2, s32);
  LLT DstTy = LLT::fixed_vector(2, s1);
  auto Three = CSEB.buildConstant(s32, 3);
  auto MinusThree = CSEB.buildConstant(s32, -3);
  auto OneOne = CSEB.buildBuildVector(VecTy, {One.getReg(0), One.getReg(0)});
  auto OneTwo = CSEB.buildBuildVector(VecTy, {One.getReg(0), Two.getReg(0)});
  auto TwoThree =
      CSEB.buildBuildVector(VecTy, {Two.getReg(0), Three.getReg(0)});
  auto MinusOneOne =
      CSEB.buildBuildVector(VecTy, {MinusOne.getReg(0), MinusOne.getReg(0)});
  auto MinusOneTwo =
      CSEB.buildBuildVector(VecTy, {MinusOne.getReg(0), MinusTwo.getReg(0)});
  auto MinusTwoThree =
      CSEB.buildBuildVector(VecTy, {MinusTwo.getReg(0), MinusThree.getReg(0)});

  // ICMP_EQ
  CSEB.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy, OneOne, OneOne);

  // ICMP_NE
  CSEB.buildICmp(CmpInst::Predicate::ICMP_NE, DstTy, OneOne, OneTwo);

  // ICMP_UGT
  CSEB.buildICmp(CmpInst::Predicate::ICMP_UGT, DstTy, TwoThree, OneTwo);

  // ICMP_UGE
  CSEB.buildICmp(CmpInst::Predicate::ICMP_UGE, DstTy, OneTwo, OneOne);

  // ICMP_ULT
  CSEB.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, OneOne, OneTwo);

  // ICMP_ULE
  CSEB.buildICmp(CmpInst::Predicate::ICMP_ULE, DstTy, OneTwo, OneOne);

  // ICMP_SGT
  CSEB.buildICmp(CmpInst::Predicate::ICMP_SGT, DstTy, MinusOneTwo,
                 MinusTwoThree);

  // ICMP_SGE
  CSEB.buildICmp(CmpInst::Predicate::ICMP_SGE, DstTy, MinusOneTwo, MinusOneOne);

  // ICMP_SLT
  CSEB.buildICmp(CmpInst::Predicate::ICMP_SLT, DstTy, MinusTwoThree,
                 MinusOneTwo);

  // ICMP_SLE
  CSEB.buildICmp(CmpInst::Predicate::ICMP_SLE, DstTy, MinusOneTwo, MinusOneOne);

  auto CheckStr = R"(
  ; CHECK: [[One:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
  ; CHECK: [[Two:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
  ; CHECK: [[MinusOne:%[0-9]+]]:_(s32) = G_CONSTANT i32 -1
  ; CHECK: [[MinusTwo:%[0-9]+]]:_(s32) = G_CONSTANT i32 -2
  ; CHECK: [[True:%[0-9]+]]:_(s1) = G_CONSTANT i1 true
  ; CHECK: [[Three:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
  ; CHECK: [[MinusThree:%[0-9]+]]:_(s32) = G_CONSTANT i32 -3
  ; CHECK: {{%[0-9]+}}:_(<2 x s32>) = G_BUILD_VECTOR [[One]]:_(s32), [[One]]:_(s32)
  ; CHECK: {{%[0-9]+}}:_(<2 x s32>) = G_BUILD_VECTOR [[One]]:_(s32), [[Two]]:_(s32)
  ; CHECK: {{%[0-9]+}}:_(<2 x s32>) = G_BUILD_VECTOR [[Two]]:_(s32), [[Three]]:_(s32)
  ; CHECK: {{%[0-9]+}}:_(<2 x s32>) = G_BUILD_VECTOR [[MinusOne]]:_(s32), [[MinusOne]]:_(s32)
  ; CHECK: {{%[0-9]+}}:_(<2 x s32>) = G_BUILD_VECTOR [[MinusOne]]:_(s32), [[MinusTwo]]:_(s32)
  ; CHECK: {{%[0-9]+}}:_(<2 x s32>) = G_BUILD_VECTOR [[MinusTwo]]:_(s32), [[MinusThree]]:_(s32)
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[True]]:_(s1), [[True]]:_(s1)
  ; CHECK: [[False:%[0-9]+]]:_(s1) = G_CONSTANT i1 false
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[False]]:_(s1), [[True]]:_(s1)
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[True]]:_(s1), [[True]]:_(s1)
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[True]]:_(s1), [[True]]:_(s1)
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[False]]:_(s1), [[True]]:_(s1)
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[True]]:_(s1), [[False]]:_(s1)
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[True]]:_(s1), [[True]]:_(s1)
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[True]]:_(s1), [[False]]:_(s1)
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[True]]:_(s1), [[True]]:_(s1)
  ; CHECK: {{%[0-9]+}}:_(<2 x s1>) = G_BUILD_VECTOR [[True]]:_(s1), [[True]]:_(s1)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

} // namespace
