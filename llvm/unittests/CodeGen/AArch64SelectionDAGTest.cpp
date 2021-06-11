//===- llvm/unittest/CodeGen/AArch64SelectionDAGTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

namespace llvm {

class AArch64SelectionDAGTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    StringRef Assembly = "define void @f() { ret void }";

    Triple TargetTriple("aarch64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    // FIXME: These tests do not depend on AArch64 specifically, but we have to
    // initialize a target. A skeleton Target for unittests would allow us to
    // always run these tests.
    if (!T)
      GTEST_SKIP();

    TargetOptions Options;
    TM = std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
        T->createTargetMachine("AArch64", "", "+sve", Options, None, None,
                               CodeGenOpt::Aggressive)));
    if (!TM)
      GTEST_SKIP();

    SMDiagnostic SMError;
    M = parseAssemblyString(Assembly, SMError, Context);
    if (!M)
      report_fatal_error(SMError.getMessage());
    M->setDataLayout(TM->createDataLayout());

    F = M->getFunction("f");
    if (!F)
      report_fatal_error("F?");

    MachineModuleInfo MMI(TM.get());

    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F), 0,
                                      MMI);

    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOpt::None);
    if (!DAG)
      report_fatal_error("DAG?");
    OptimizationRemarkEmitter ORE(F);
    DAG->init(*MF, ORE, nullptr, nullptr, nullptr, nullptr, nullptr);
  }

  TargetLoweringBase::LegalizeTypeAction getTypeAction(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeAction(Context, VT);
  }

  EVT getTypeToTransformTo(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeToTransformTo(Context, VT);
  }

  LLVMContext Context;
  std::unique_ptr<LLVMTargetMachine> TM;
  std::unique_ptr<Module> M;
  Function *F;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

TEST_F(AArch64SelectionDAGTest, computeKnownBits_ZERO_EXTEND_VECTOR_INREG) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto Int16VT = EVT::getIntegerVT(Context, 16);
  auto InVecVT = EVT::getVectorVT(Context, Int8VT, 4);
  auto OutVecVT = EVT::getVectorVT(Context, Int16VT, 2);
  auto InVec = DAG->getConstant(0, Loc, InVecVT);
  auto Op = DAG->getNode(ISD::ZERO_EXTEND_VECTOR_INREG, Loc, OutVecVT, InVec);
  auto DemandedElts = APInt(2, 3);
  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);
  EXPECT_TRUE(Known.isZero());
}

TEST_F(AArch64SelectionDAGTest, computeKnownBitsSVE_ZERO_EXTEND_VECTOR_INREG) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto Int16VT = EVT::getIntegerVT(Context, 16);
  auto InVecVT = EVT::getVectorVT(Context, Int8VT, 4, true);
  auto OutVecVT = EVT::getVectorVT(Context, Int16VT, 2, true);
  auto InVec = DAG->getConstant(0, Loc, InVecVT);
  auto Op = DAG->getNode(ISD::ZERO_EXTEND_VECTOR_INREG, Loc, OutVecVT, InVec);
  auto DemandedElts = APInt(2, 3);
  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);

  // We don't know anything for SVE at the moment.
  EXPECT_EQ(Known.Zero, APInt(16, 0u));
  EXPECT_EQ(Known.One, APInt(16, 0u));
  EXPECT_FALSE(Known.isZero());
}

TEST_F(AArch64SelectionDAGTest, computeKnownBits_EXTRACT_SUBVECTOR) {
  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 3);
  auto IdxVT = EVT::getIntegerVT(Context, 64);
  auto Vec = DAG->getConstant(0, Loc, VecVT);
  auto ZeroIdx = DAG->getConstant(0, Loc, IdxVT);
  auto Op = DAG->getNode(ISD::EXTRACT_SUBVECTOR, Loc, VecVT, Vec, ZeroIdx);
  auto DemandedElts = APInt(3, 7);
  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);
  EXPECT_TRUE(Known.isZero());
}

TEST_F(AArch64SelectionDAGTest, ComputeNumSignBits_SIGN_EXTEND_VECTOR_INREG) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto Int16VT = EVT::getIntegerVT(Context, 16);
  auto InVecVT = EVT::getVectorVT(Context, Int8VT, 4);
  auto OutVecVT = EVT::getVectorVT(Context, Int16VT, 2);
  auto InVec = DAG->getConstant(1, Loc, InVecVT);
  auto Op = DAG->getNode(ISD::SIGN_EXTEND_VECTOR_INREG, Loc, OutVecVT, InVec);
  auto DemandedElts = APInt(2, 3);
  EXPECT_EQ(DAG->ComputeNumSignBits(Op, DemandedElts), 15u);
}

TEST_F(AArch64SelectionDAGTest, ComputeNumSignBitsSVE_SIGN_EXTEND_VECTOR_INREG) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto Int16VT = EVT::getIntegerVT(Context, 16);
  auto InVecVT = EVT::getVectorVT(Context, Int8VT, 4, /*IsScalable=*/true);
  auto OutVecVT = EVT::getVectorVT(Context, Int16VT, 2, /*IsScalable=*/true);
  auto InVec = DAG->getConstant(1, Loc, InVecVT);
  auto Op = DAG->getNode(ISD::SIGN_EXTEND_VECTOR_INREG, Loc, OutVecVT, InVec);
  auto DemandedElts = APInt(2, 3);
  EXPECT_EQ(DAG->ComputeNumSignBits(Op, DemandedElts), 1u);
}

TEST_F(AArch64SelectionDAGTest, ComputeNumSignBits_EXTRACT_SUBVECTOR) {
  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 3);
  auto IdxVT = EVT::getIntegerVT(Context, 64);
  auto Vec = DAG->getConstant(1, Loc, VecVT);
  auto ZeroIdx = DAG->getConstant(0, Loc, IdxVT);
  auto Op = DAG->getNode(ISD::EXTRACT_SUBVECTOR, Loc, VecVT, Vec, ZeroIdx);
  auto DemandedElts = APInt(3, 7);
  EXPECT_EQ(DAG->ComputeNumSignBits(Op, DemandedElts), 7u);
}

TEST_F(AArch64SelectionDAGTest, SimplifyDemandedVectorElts_EXTRACT_SUBVECTOR) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 3);
  auto IdxVT = EVT::getIntegerVT(Context, 64);
  auto Vec = DAG->getConstant(1, Loc, VecVT);
  auto ZeroIdx = DAG->getConstant(0, Loc, IdxVT);
  auto Op = DAG->getNode(ISD::EXTRACT_SUBVECTOR, Loc, VecVT, Vec, ZeroIdx);
  auto DemandedElts = APInt(3, 7);
  auto KnownUndef = APInt(3, 0);
  auto KnownZero = APInt(3, 0);
  TargetLowering::TargetLoweringOpt TLO(*DAG, false, false);
  EXPECT_EQ(TL.SimplifyDemandedVectorElts(Op, DemandedElts, KnownUndef,
                                          KnownZero, TLO),
            false);
}

TEST_F(AArch64SelectionDAGTest, SimplifyDemandedBitsNEON) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto InVecVT = EVT::getVectorVT(Context, Int8VT, 16);
  SDValue UnknownOp = DAG->getRegister(0, InVecVT);
  SDValue Mask1S = DAG->getConstant(0x8A, Loc, Int8VT);
  SDValue Mask1V = DAG->getSplatBuildVector(InVecVT, Loc, Mask1S);
  SDValue N0 = DAG->getNode(ISD::AND, Loc, InVecVT, Mask1V, UnknownOp);

  SDValue Mask2S = DAG->getConstant(0x55, Loc, Int8VT);
  SDValue Mask2V = DAG->getSplatBuildVector(InVecVT, Loc, Mask2S);

  SDValue Op = DAG->getNode(ISD::AND, Loc, InVecVT, N0, Mask2V);
  // N0 = ?000?0?0
  // Mask2V = 01010101
  //  =>
  // Known.Zero = 00100000 (0xAA)
  KnownBits Known;
  APInt DemandedBits = APInt(8, 0xFF);
  TargetLowering::TargetLoweringOpt TLO(*DAG, false, false);
  EXPECT_TRUE(TL.SimplifyDemandedBits(Op, DemandedBits, Known, TLO));
  EXPECT_EQ(Known.Zero, APInt(8, 0xAA));
}

TEST_F(AArch64SelectionDAGTest, SimplifyDemandedBitsSVE) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto InVecVT = EVT::getVectorVT(Context, Int8VT, 16, /*IsScalable=*/true);
  SDValue UnknownOp = DAG->getRegister(0, InVecVT);
  SDValue Mask1S = DAG->getConstant(0x8A, Loc, Int8VT);
  SDValue Mask1V = DAG->getSplatVector(InVecVT, Loc, Mask1S);
  SDValue N0 = DAG->getNode(ISD::AND, Loc, InVecVT, Mask1V, UnknownOp);

  SDValue Mask2S = DAG->getConstant(0x55, Loc, Int8VT);
  SDValue Mask2V = DAG->getSplatVector(InVecVT, Loc, Mask2S);

  SDValue Op = DAG->getNode(ISD::AND, Loc, InVecVT, N0, Mask2V);

  KnownBits Known;
  APInt DemandedBits = APInt(8, 0xFF);
  TargetLowering::TargetLoweringOpt TLO(*DAG, false, false);
  EXPECT_FALSE(TL.SimplifyDemandedBits(Op, DemandedBits, Known, TLO));
  EXPECT_EQ(Known.Zero, APInt(8, 0));
}

// Piggy-backing on the AArch64 tests to verify SelectionDAG::computeKnownBits.
TEST_F(AArch64SelectionDAGTest, ComputeKnownBits_ADD) {
  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto UnknownOp = DAG->getRegister(0, IntVT);
  auto Mask = DAG->getConstant(0x8A, Loc, IntVT);
  auto N0 = DAG->getNode(ISD::AND, Loc, IntVT, Mask, UnknownOp);
  auto N1 = DAG->getConstant(0x55, Loc, IntVT);
  auto Op = DAG->getNode(ISD::ADD, Loc, IntVT, N0, N1);
  // N0 = ?000?0?0
  // N1 = 01010101
  //  =>
  // Known.One  = 01010101 (0x55)
  // Known.Zero = 00100000 (0x20)
  KnownBits Known = DAG->computeKnownBits(Op);
  EXPECT_EQ(Known.Zero, APInt(8, 0x20));
  EXPECT_EQ(Known.One, APInt(8, 0x55));
}

// Piggy-backing on the AArch64 tests to verify SelectionDAG::computeKnownBits.
TEST_F(AArch64SelectionDAGTest, ComputeKnownBits_SUB) {
  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto N0 = DAG->getConstant(0x55, Loc, IntVT);
  auto UnknownOp = DAG->getRegister(0, IntVT);
  auto Mask = DAG->getConstant(0x2e, Loc, IntVT);
  auto N1 = DAG->getNode(ISD::AND, Loc, IntVT, Mask, UnknownOp);
  auto Op = DAG->getNode(ISD::SUB, Loc, IntVT, N0, N1);
  // N0 = 01010101
  // N1 = 00?0???0
  //  =>
  // Known.One  = 00000001 (0x1)
  // Known.Zero = 10000000 (0x80)
  KnownBits Known = DAG->computeKnownBits(Op);
  EXPECT_EQ(Known.Zero, APInt(8, 0x80));
  EXPECT_EQ(Known.One, APInt(8, 0x1));
}

TEST_F(AArch64SelectionDAGTest, isSplatValue_Fixed_BUILD_VECTOR) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 16, false);
  // Create a BUILD_VECTOR
  SDValue Op = DAG->getConstant(1, Loc, VecVT);
  EXPECT_EQ(Op->getOpcode(), ISD::BUILD_VECTOR);
  EXPECT_TRUE(DAG->isSplatValue(Op, /*AllowUndefs=*/false));

  APInt UndefElts;
  APInt DemandedElts;
  EXPECT_FALSE(DAG->isSplatValue(Op, DemandedElts, UndefElts));

  // Width=16, Mask=3
  DemandedElts = APInt(16, 3);
  EXPECT_TRUE(DAG->isSplatValue(Op, DemandedElts, UndefElts));
}

TEST_F(AArch64SelectionDAGTest, isSplatValue_Fixed_ADD_of_BUILD_VECTOR) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 16, false);

  // Should create BUILD_VECTORs
  SDValue Val1 = DAG->getConstant(1, Loc, VecVT);
  SDValue Val2 = DAG->getConstant(3, Loc, VecVT);
  EXPECT_EQ(Val1->getOpcode(), ISD::BUILD_VECTOR);
  SDValue Op = DAG->getNode(ISD::ADD, Loc, VecVT, Val1, Val2);

  EXPECT_TRUE(DAG->isSplatValue(Op, /*AllowUndefs=*/false));

  APInt UndefElts;
  APInt DemandedElts;
  EXPECT_FALSE(DAG->isSplatValue(Op, DemandedElts, UndefElts));

  // Width=16, Mask=3
  DemandedElts = APInt(16, 3);
  EXPECT_TRUE(DAG->isSplatValue(Op, DemandedElts, UndefElts));
}

TEST_F(AArch64SelectionDAGTest, isSplatValue_Scalable_SPLAT_VECTOR) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 16, true);
  // Create a SPLAT_VECTOR
  SDValue Op = DAG->getConstant(1, Loc, VecVT);
  EXPECT_EQ(Op->getOpcode(), ISD::SPLAT_VECTOR);
  EXPECT_TRUE(DAG->isSplatValue(Op, /*AllowUndefs=*/false));

  APInt UndefElts;
  APInt DemandedElts;
  EXPECT_TRUE(DAG->isSplatValue(Op, DemandedElts, UndefElts));

  // Width=16, Mask=3. These bits should be ignored.
  DemandedElts = APInt(16, 3);
  EXPECT_TRUE(DAG->isSplatValue(Op, DemandedElts, UndefElts));
}

TEST_F(AArch64SelectionDAGTest, isSplatValue_Scalable_ADD_of_SPLAT_VECTOR) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 16, true);

  // Should create SPLAT_VECTORS
  SDValue Val1 = DAG->getConstant(1, Loc, VecVT);
  SDValue Val2 = DAG->getConstant(3, Loc, VecVT);
  EXPECT_EQ(Val1->getOpcode(), ISD::SPLAT_VECTOR);
  SDValue Op = DAG->getNode(ISD::ADD, Loc, VecVT, Val1, Val2);

  EXPECT_TRUE(DAG->isSplatValue(Op, /*AllowUndefs=*/false));

  APInt UndefElts;
  APInt DemandedElts;
  EXPECT_TRUE(DAG->isSplatValue(Op, DemandedElts, UndefElts));

  // Width=16, Mask=3. These bits should be ignored.
  DemandedElts = APInt(16, 3);
  EXPECT_TRUE(DAG->isSplatValue(Op, DemandedElts, UndefElts));
}

TEST_F(AArch64SelectionDAGTest, getSplatSourceVector_Fixed_BUILD_VECTOR) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 16, false);
  // Create a BUILD_VECTOR
  SDValue Op = DAG->getConstant(1, Loc, VecVT);
  EXPECT_EQ(Op->getOpcode(), ISD::BUILD_VECTOR);

  int SplatIdx = -1;
  EXPECT_EQ(DAG->getSplatSourceVector(Op, SplatIdx), Op);
  EXPECT_EQ(SplatIdx, 0);
}

TEST_F(AArch64SelectionDAGTest, getSplatSourceVector_Fixed_ADD_of_BUILD_VECTOR) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 16, false);

  // Should create BUILD_VECTORs
  SDValue Val1 = DAG->getConstant(1, Loc, VecVT);
  SDValue Val2 = DAG->getConstant(3, Loc, VecVT);
  EXPECT_EQ(Val1->getOpcode(), ISD::BUILD_VECTOR);
  SDValue Op = DAG->getNode(ISD::ADD, Loc, VecVT, Val1, Val2);

  int SplatIdx = -1;
  EXPECT_EQ(DAG->getSplatSourceVector(Op, SplatIdx), Op);
  EXPECT_EQ(SplatIdx, 0);
}

TEST_F(AArch64SelectionDAGTest, getSplatSourceVector_Scalable_SPLAT_VECTOR) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 16, true);
  // Create a SPLAT_VECTOR
  SDValue Op = DAG->getConstant(1, Loc, VecVT);
  EXPECT_EQ(Op->getOpcode(), ISD::SPLAT_VECTOR);

  int SplatIdx = -1;
  EXPECT_EQ(DAG->getSplatSourceVector(Op, SplatIdx), Op);
  EXPECT_EQ(SplatIdx, 0);
}

TEST_F(AArch64SelectionDAGTest, getSplatSourceVector_Scalable_ADD_of_SPLAT_VECTOR) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, IntVT, 16, true);

  // Should create SPLAT_VECTORS
  SDValue Val1 = DAG->getConstant(1, Loc, VecVT);
  SDValue Val2 = DAG->getConstant(3, Loc, VecVT);
  EXPECT_EQ(Val1->getOpcode(), ISD::SPLAT_VECTOR);
  SDValue Op = DAG->getNode(ISD::ADD, Loc, VecVT, Val1, Val2);

  int SplatIdx = -1;
  EXPECT_EQ(DAG->getSplatSourceVector(Op, SplatIdx), Op);
  EXPECT_EQ(SplatIdx, 0);
}

TEST_F(AArch64SelectionDAGTest, getRepeatedSequence_Patterns) {
  TargetLowering TL(*TM);

  SDLoc Loc;
  unsigned NumElts = 16;
  MVT IntVT = MVT::i8;
  MVT VecVT = MVT::getVectorVT(IntVT, NumElts);

  // Base scalar constants.
  SDValue Val0 = DAG->getConstant(0, Loc, IntVT);
  SDValue Val1 = DAG->getConstant(1, Loc, IntVT);
  SDValue Val2 = DAG->getConstant(2, Loc, IntVT);
  SDValue Val3 = DAG->getConstant(3, Loc, IntVT);
  SDValue UndefVal = DAG->getUNDEF(IntVT);

  // Build some repeating sequences.
  SmallVector<SDValue, 16> Pattern1111, Pattern1133, Pattern0123;
  for(int I = 0; I != 4; ++I) {
    Pattern1111.append(4, Val1);
    Pattern1133.append(2, Val1);
    Pattern1133.append(2, Val3);
    Pattern0123.push_back(Val0);
    Pattern0123.push_back(Val1);
    Pattern0123.push_back(Val2);
    Pattern0123.push_back(Val3);
  }

  // Build a non-pow2 repeating sequence.
  SmallVector<SDValue, 16> Pattern022;
  Pattern022.push_back(Val0);
  Pattern022.append(2, Val2);
  Pattern022.push_back(Val0);
  Pattern022.append(2, Val2);
  Pattern022.push_back(Val0);
  Pattern022.append(2, Val2);
  Pattern022.push_back(Val0);
  Pattern022.append(2, Val2);
  Pattern022.push_back(Val0);
  Pattern022.append(2, Val2);
  Pattern022.push_back(Val0);

  // Build a non-repeating sequence.
  SmallVector<SDValue, 16> Pattern1_3;
  Pattern1_3.append(8, Val1);
  Pattern1_3.append(8, Val3);

  // Add some undefs to make it trickier.
  Pattern1111[1] = Pattern1111[2] = Pattern1111[15] = UndefVal;
  Pattern1133[0] = Pattern1133[2] = UndefVal;

  auto *BV1111 =
      cast<BuildVectorSDNode>(DAG->getBuildVector(VecVT, Loc, Pattern1111));
  auto *BV1133 =
      cast<BuildVectorSDNode>(DAG->getBuildVector(VecVT, Loc, Pattern1133));
  auto *BV0123=
      cast<BuildVectorSDNode>(DAG->getBuildVector(VecVT, Loc, Pattern0123));
  auto *BV022 =
      cast<BuildVectorSDNode>(DAG->getBuildVector(VecVT, Loc, Pattern022));
  auto *BV1_3 =
      cast<BuildVectorSDNode>(DAG->getBuildVector(VecVT, Loc, Pattern1_3));

  // Check for sequences.
  SmallVector<SDValue, 16> Seq1111, Seq1133, Seq0123, Seq022, Seq1_3;
  BitVector Undefs1111, Undefs1133, Undefs0123, Undefs022, Undefs1_3;

  EXPECT_TRUE(BV1111->getRepeatedSequence(Seq1111, &Undefs1111));
  EXPECT_EQ(Undefs1111.count(), 3u);
  EXPECT_EQ(Seq1111.size(), 1u);
  EXPECT_EQ(Seq1111[0], Val1);

  EXPECT_TRUE(BV1133->getRepeatedSequence(Seq1133, &Undefs1133));
  EXPECT_EQ(Undefs1133.count(), 2u);
  EXPECT_EQ(Seq1133.size(), 4u);
  EXPECT_EQ(Seq1133[0], Val1);
  EXPECT_EQ(Seq1133[1], Val1);
  EXPECT_EQ(Seq1133[2], Val3);
  EXPECT_EQ(Seq1133[3], Val3);

  EXPECT_TRUE(BV0123->getRepeatedSequence(Seq0123, &Undefs0123));
  EXPECT_EQ(Undefs0123.count(), 0u);
  EXPECT_EQ(Seq0123.size(), 4u);
  EXPECT_EQ(Seq0123[0], Val0);
  EXPECT_EQ(Seq0123[1], Val1);
  EXPECT_EQ(Seq0123[2], Val2);
  EXPECT_EQ(Seq0123[3], Val3);

  EXPECT_FALSE(BV022->getRepeatedSequence(Seq022, &Undefs022));
  EXPECT_FALSE(BV1_3->getRepeatedSequence(Seq1_3, &Undefs1_3));

  // Try again with DemandedElts masks.
  APInt Mask1111_0 = APInt::getOneBitSet(NumElts, 0);
  EXPECT_TRUE(BV1111->getRepeatedSequence(Mask1111_0, Seq1111, &Undefs1111));
  EXPECT_EQ(Undefs1111.count(), 0u);
  EXPECT_EQ(Seq1111.size(), 1u);
  EXPECT_EQ(Seq1111[0], Val1);

  APInt Mask1111_1 = APInt::getOneBitSet(NumElts, 2);
  EXPECT_TRUE(BV1111->getRepeatedSequence(Mask1111_1, Seq1111, &Undefs1111));
  EXPECT_EQ(Undefs1111.count(), 1u);
  EXPECT_EQ(Seq1111.size(), 1u);
  EXPECT_EQ(Seq1111[0], UndefVal);

  APInt Mask0123 = APInt(NumElts, 0x7777);
  EXPECT_TRUE(BV0123->getRepeatedSequence(Mask0123, Seq0123, &Undefs0123));
  EXPECT_EQ(Undefs0123.count(), 0u);
  EXPECT_EQ(Seq0123.size(), 4u);
  EXPECT_EQ(Seq0123[0], Val0);
  EXPECT_EQ(Seq0123[1], Val1);
  EXPECT_EQ(Seq0123[2], Val2);
  EXPECT_EQ(Seq0123[3], SDValue());

  APInt Mask1_3 = APInt::getHighBitsSet(16, 8);
  EXPECT_TRUE(BV1_3->getRepeatedSequence(Mask1_3, Seq1_3, &Undefs1_3));
  EXPECT_EQ(Undefs1_3.count(), 0u);
  EXPECT_EQ(Seq1_3.size(), 1u);
  EXPECT_EQ(Seq1_3[0], Val3);
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_SplitScalableMVT) {
  MVT VT = MVT::nxv4i64;
  EXPECT_EQ(getTypeAction(VT), TargetLoweringBase::TypeSplitVector);
  ASSERT_TRUE(getTypeToTransformTo(VT).isScalableVector());
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_PromoteScalableMVT) {
  MVT VT = MVT::nxv2i32;
  EXPECT_EQ(getTypeAction(VT), TargetLoweringBase::TypePromoteInteger);
  ASSERT_TRUE(getTypeToTransformTo(VT).isScalableVector());
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_NoScalarizeMVT_nxv1f32) {
  MVT VT = MVT::nxv1f32;
  EXPECT_NE(getTypeAction(VT), TargetLoweringBase::TypeScalarizeVector);
  ASSERT_TRUE(getTypeToTransformTo(VT).isScalableVector());
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_SplitScalableEVT) {
  EVT VT = EVT::getVectorVT(Context, MVT::i64, 256, true);
  EXPECT_EQ(getTypeAction(VT), TargetLoweringBase::TypeSplitVector);
  EXPECT_EQ(getTypeToTransformTo(VT), VT.getHalfNumVectorElementsVT(Context));
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_WidenScalableEVT) {
  EVT FromVT = EVT::getVectorVT(Context, MVT::i64, 6, true);
  EVT ToVT = EVT::getVectorVT(Context, MVT::i64, 8, true);

  EXPECT_EQ(getTypeAction(FromVT), TargetLoweringBase::TypeWidenVector);
  EXPECT_EQ(getTypeToTransformTo(FromVT), ToVT);
}

TEST_F(AArch64SelectionDAGTest,
       getTypeConversion_ScalarizeScalableEVT_nxv1f128) {
  EVT VT = EVT::getVectorVT(Context, MVT::f128, ElementCount::getScalable(1));
  EXPECT_EQ(getTypeAction(VT), TargetLoweringBase::TypeScalarizeScalableVector);
  EXPECT_EQ(getTypeToTransformTo(VT), MVT::f128);
}

TEST_F(AArch64SelectionDAGTest, TestFold_STEP_VECTOR) {
  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, MVT::i8, 16, true);

  // Should create SPLAT_VECTOR
  SDValue Zero = DAG->getConstant(0, Loc, IntVT);
  SDValue Op = DAG->getNode(ISD::STEP_VECTOR, Loc, VecVT, Zero);
  EXPECT_EQ(Op.getOpcode(), ISD::SPLAT_VECTOR);
}

} // end namespace llvm
