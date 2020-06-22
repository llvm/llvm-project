//===- llvm/unittest/CodeGen/AArch64SelectionDAGTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
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
      return;

    TargetOptions Options;
    TM = std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
        T->createTargetMachine("AArch64", "", "+sve", Options, None, None,
                               CodeGenOpt::Aggressive)));
    if (!TM)
      return;

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
  if (!TM)
    return;
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
  if (!TM)
    return;
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
  if (!TM)
    return;
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
  if (!TM)
    return;
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
  if (!TM)
    return;
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
  if (!TM)
    return;
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
  if (!TM)
    return;

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
  if (!TM)
    return;

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
  if (!TM)
    return;

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
  if (!TM)
    return;
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
  if (!TM)
    return;
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
  if (!TM)
    return;

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
  if (!TM)
    return;

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
  if (!TM)
    return;

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
  if (!TM)
    return;

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
  if (!TM)
    return;

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
  if (!TM)
    return;

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
  if (!TM)
    return;

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
  if (!TM)
    return;

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

TEST_F(AArch64SelectionDAGTest, getTypeConversion_SplitScalableMVT) {
  if (!TM)
    return;

  MVT VT = MVT::nxv4i64;
  EXPECT_EQ(getTypeAction(VT), TargetLoweringBase::TypeSplitVector);
  ASSERT_TRUE(getTypeToTransformTo(VT).isScalableVector());
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_PromoteScalableMVT) {
  if (!TM)
    return;

  MVT VT = MVT::nxv2i32;
  EXPECT_EQ(getTypeAction(VT), TargetLoweringBase::TypePromoteInteger);
  ASSERT_TRUE(getTypeToTransformTo(VT).isScalableVector());
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_NoScalarizeMVT_nxv1f32) {
  if (!TM)
    return;

  MVT VT = MVT::nxv1f32;
  EXPECT_NE(getTypeAction(VT), TargetLoweringBase::TypeScalarizeVector);
  ASSERT_TRUE(getTypeToTransformTo(VT).isScalableVector());
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_SplitScalableEVT) {
  if (!TM)
    return;

  EVT VT = EVT::getVectorVT(Context, MVT::i64, 256, true);
  EXPECT_EQ(getTypeAction(VT), TargetLoweringBase::TypeSplitVector);
  EXPECT_EQ(getTypeToTransformTo(VT), VT.getHalfNumVectorElementsVT(Context));
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_WidenScalableEVT) {
  if (!TM)
    return;

  EVT FromVT = EVT::getVectorVT(Context, MVT::i64, 6, true);
  EVT ToVT = EVT::getVectorVT(Context, MVT::i64, 8, true);

  EXPECT_EQ(getTypeAction(FromVT), TargetLoweringBase::TypeWidenVector);
  EXPECT_EQ(getTypeToTransformTo(FromVT), ToVT);
}

TEST_F(AArch64SelectionDAGTest, getTypeConversion_NoScalarizeEVT_nxv1f128) {
  if (!TM)
    return;

  EVT FromVT = EVT::getVectorVT(Context, MVT::f128, 1, true);
  EXPECT_DEATH(getTypeAction(FromVT), "Cannot legalize this vector");
}

} // end namespace llvm
