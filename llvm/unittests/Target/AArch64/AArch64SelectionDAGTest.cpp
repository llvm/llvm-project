//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64SelectionDAGInfo.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

namespace llvm {

class AArch64SelectionDAGTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
  }

  void SetUp() override {
    StringRef Assembly = "define void @f() { ret void }";

    Triple TargetTriple("aarch64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(
        T->createTargetMachine(TargetTriple, "", "+sve", Options, std::nullopt,
                               std::nullopt, CodeGenOptLevel::Aggressive));

    SMDiagnostic SMError;
    M = parseAssemblyString(Assembly, SMError, Context);
    if (!M)
      report_fatal_error(SMError.getMessage());
    M->setDataLayout(TM->createDataLayout());

    F = M->getFunction("f");
    if (!F)
      report_fatal_error("F?");

    MachineModuleInfo MMI(TM.get());

    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           MMI.getContext(), 0);

    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOptLevel::None);
    if (!DAG)
      report_fatal_error("DAG?");
    OptimizationRemarkEmitter ORE(F);
    DAG->init(*MF, ORE, nullptr, nullptr, nullptr, nullptr, nullptr, MMI,
              nullptr);
  }

  TargetLoweringBase::LegalizeTypeAction getTypeAction(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeAction(Context, VT);
  }

  EVT getTypeToTransformTo(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeToTransformTo(Context, VT);
  }

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
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

TEST_F(AArch64SelectionDAGTest,
       ComputeNumSignBitsSVE_SIGN_EXTEND_VECTOR_INREG) {
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

TEST_F(AArch64SelectionDAGTest, ComputeNumSignBits_VASHR) {
  SDLoc Loc;
  auto VecVT = MVT::v8i8;
  auto Shift = DAG->getConstant(4, Loc, MVT::i32);
  auto Vec0 = DAG->getConstant(1, Loc, VecVT);
  auto Op1 = DAG->getNode(AArch64ISD::VASHR, Loc, VecVT, Vec0, Shift);
  EXPECT_EQ(DAG->ComputeNumSignBits(Op1), 8u);
  auto VecA = DAG->getConstant(0xaa, Loc, VecVT);
  auto Op2 = DAG->getNode(AArch64ISD::VASHR, Loc, VecVT, VecA, Shift);
  EXPECT_EQ(DAG->ComputeNumSignBits(Op2), 5u);
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
TEST_F(AArch64SelectionDAGTest, ComputeKnownBits_UADDO_CARRY) {
  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto UnknownOp = DAG->getRegister(0, IntVT);
  auto Mask_Zero = DAG->getConstant(0x28, Loc, IntVT);
  auto Mask_One = DAG->getConstant(0x20, Loc, IntVT);
  auto N0 = DAG->getNode(ISD::AND, Loc, IntVT, Mask_Zero, UnknownOp);
  N0 = DAG->getNode(ISD::OR, Loc, IntVT, Mask_One, N0);
  auto N1 = DAG->getConstant(0x65, Loc, IntVT);

  KnownBits Known;

  auto UnknownBorrow = DAG->getRegister(1, IntVT);
  auto OpUnknownBorrow =
      DAG->getNode(ISD::UADDO_CARRY, Loc, IntVT, N0, N1, UnknownBorrow);
  // N0 = 0010?000
  // N1 = 01100101
  // B  =        ?
  //  =>
  // Known.Zero = 01110000 (0x70)
  // Known.One  = 10000100 (0x84)
  Known = DAG->computeKnownBits(OpUnknownBorrow);
  EXPECT_EQ(Known.Zero, APInt(8, 0x70));
  EXPECT_EQ(Known.One, APInt(8, 0x84));

  auto ZeroBorrow = DAG->getConstant(0x0, Loc, IntVT);
  auto OpZeroBorrow =
      DAG->getNode(ISD::UADDO_CARRY, Loc, IntVT, N0, N1, ZeroBorrow);
  // N0 = 0010?000
  // N1 = 01100101
  // B  =        0
  //  =>
  // Known.Zero = 01110010 (0x72)
  // Known.One  = 10000101 (0x85)
  Known = DAG->computeKnownBits(OpZeroBorrow);
  EXPECT_EQ(Known.Zero, APInt(8, 0x72));
  EXPECT_EQ(Known.One, APInt(8, 0x85));

  auto OneBorrow = DAG->getConstant(0x1, Loc, IntVT);
  auto OpOneBorrow =
      DAG->getNode(ISD::UADDO_CARRY, Loc, IntVT, N0, N1, OneBorrow);
  // N0 = 0010?000
  // N1 = 01100101
  // B  =        1
  //  =>
  // Known.Zero = 01110001 (0x71)
  // Known.One  = 10000110 (0x86)
  Known = DAG->computeKnownBits(OpOneBorrow);
  EXPECT_EQ(Known.Zero, APInt(8, 0x71));
  EXPECT_EQ(Known.One, APInt(8, 0x86));
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

// Piggy-backing on the AArch64 tests to verify SelectionDAG::computeKnownBits.
TEST_F(AArch64SelectionDAGTest, ComputeKnownBits_USUBO_CARRY) {
  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 8);
  auto N0 = DAG->getConstant(0x5a, Loc, IntVT);
  auto UnknownOp = DAG->getRegister(0, IntVT);         // ????????
  auto Mask1_Zero = DAG->getConstant(0x8, Loc, IntVT); // 00001000
  auto Mask1_One = DAG->getConstant(0x20, Loc, IntVT); // 00100000
  // N1 = (???????? & 00001000) | 00100000 = 0010?000
  auto N1 = DAG->getNode(ISD::AND, Loc, IntVT, Mask1_Zero, UnknownOp);
  N1 = DAG->getNode(ISD::OR, Loc, IntVT, Mask1_One, N1);

  KnownBits Known;

  auto UnknownBorrow = DAG->getRegister(1, IntVT);
  auto OpUnknownBorrow =
      DAG->getNode(ISD::USUBO_CARRY, Loc, IntVT, N0, N1, UnknownBorrow);
  // N0 = 01011010
  // N1 = 0010?000
  // B  =        ?
  //  =>
  // Known.Zero = 11000100 (0xc4)
  // Known.One  = 00110000 (0x30)
  Known = DAG->computeKnownBits(OpUnknownBorrow);
  EXPECT_EQ(Known.Zero, APInt(8, 0xc4));
  EXPECT_EQ(Known.One, APInt(8, 0x30));

  auto ZeroBorrow = DAG->getConstant(0x0, Loc, IntVT);
  auto OpZeroBorrow =
      DAG->getNode(ISD::USUBO_CARRY, Loc, IntVT, N0, N1, ZeroBorrow);
  // N0 = 01011010
  // N1 = 0010?000
  // B  =        0
  //  =>
  // Known.Zero = 11000101 (0xc5)
  // Known.One  = 00110010 (0x32)
  Known = DAG->computeKnownBits(OpZeroBorrow);
  EXPECT_EQ(Known.Zero, APInt(8, 0xc5));
  EXPECT_EQ(Known.One, APInt(8, 0x32));

  auto OneBorrow = DAG->getConstant(0x1, Loc, IntVT);
  auto OpOneBorrow =
      DAG->getNode(ISD::USUBO_CARRY, Loc, IntVT, N0, N1, OneBorrow);
  // N0 = 01011010
  // N1 = 0010?000
  // B  =        1
  //  =>
  // Known.Zero = 11000110 (0xc6)
  // Known.One  = 00110001 (0x31)
  Known = DAG->computeKnownBits(OpOneBorrow);
  EXPECT_EQ(Known.Zero, APInt(8, 0xc6));
  EXPECT_EQ(Known.One, APInt(8, 0x31));
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
  APInt DemandedElts(1, 1);
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
  APInt DemandedElts(1, 1);
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

TEST_F(AArch64SelectionDAGTest,
       getSplatSourceVector_Fixed_ADD_of_BUILD_VECTOR) {
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

TEST_F(AArch64SelectionDAGTest,
       getSplatSourceVector_Scalable_ADD_of_SPLAT_VECTOR) {
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
  for (int I = 0; I != 4; ++I) {
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
  auto *BV0123 =
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

TEST_F(AArch64SelectionDAGTest, ReplaceAllUsesWith) {
  SDLoc Loc;
  EVT IntVT = EVT::getIntegerVT(Context, 8);

  SDValue N0 = DAG->getConstant(0x42, Loc, IntVT);
  SDValue N1 = DAG->getRegister(0, IntVT);
  // Construct node to fill arbitrary ExtraInfo.
  SDValue N2 = DAG->getNode(ISD::SUB, Loc, IntVT, N0, N1);
  EXPECT_FALSE(DAG->getHeapAllocSite(N2.getNode()));
  EXPECT_FALSE(DAG->getNoMergeSiteInfo(N2.getNode()));
  EXPECT_FALSE(DAG->getPCSections(N2.getNode()));
  MDNode *MD = MDNode::get(Context, {});
  DAG->addHeapAllocSite(N2.getNode(), MD);
  DAG->addNoMergeSiteInfo(N2.getNode(), true);
  DAG->addPCSections(N2.getNode(), MD);
  EXPECT_EQ(DAG->getHeapAllocSite(N2.getNode()), MD);
  EXPECT_TRUE(DAG->getNoMergeSiteInfo(N2.getNode()));
  EXPECT_EQ(DAG->getPCSections(N2.getNode()), MD);

  SDValue Root = DAG->getNode(ISD::ADD, Loc, IntVT, N2, N2);
  EXPECT_EQ(Root->getOperand(0)->getOpcode(), ISD::SUB);
  // Create new node and check that ExtraInfo is propagated on RAUW.
  SDValue New = DAG->getNode(ISD::ADD, Loc, IntVT, N1, N1);
  EXPECT_FALSE(DAG->getHeapAllocSite(New.getNode()));
  EXPECT_FALSE(DAG->getNoMergeSiteInfo(New.getNode()));
  EXPECT_FALSE(DAG->getPCSections(New.getNode()));

  DAG->ReplaceAllUsesWith(N2, New);
  EXPECT_EQ(Root->getOperand(0), New);
  EXPECT_EQ(DAG->getHeapAllocSite(New.getNode()), MD);
  EXPECT_TRUE(DAG->getNoMergeSiteInfo(New.getNode()));
  EXPECT_EQ(DAG->getPCSections(New.getNode()), MD);
}

TEST_F(AArch64SelectionDAGTest, computeKnownBits_extload_known01) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto Int64VT = EVT::getIntegerVT(Context, 64);
  auto Ptr = DAG->getConstant(0, Loc, Int64VT);
  auto PtrInfo =
      MachinePointerInfo::getFixedStack(DAG->getMachineFunction(), 0);
  AAMDNodes AA;
  MDBuilder MDHelper(*DAG->getContext());
  MDNode *Range = MDHelper.createRange(APInt(8, 0), APInt(8, 2));
  MachineMemOperand *MMO = DAG->getMachineFunction().getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOLoad, 8, Align(8), AA, Range);

  auto ALoad = DAG->getExtLoad(ISD::EXTLOAD, Loc, Int32VT, DAG->getEntryNode(),
                               Ptr, Int8VT, MMO);
  KnownBits Known = DAG->computeKnownBits(ALoad);
  EXPECT_EQ(Known.Zero, APInt(32, 0xfe));
  EXPECT_EQ(Known.One, APInt(32, 0));

  auto ZLoad = DAG->getExtLoad(ISD::ZEXTLOAD, Loc, Int32VT, DAG->getEntryNode(),
                               Ptr, Int8VT, MMO);
  Known = DAG->computeKnownBits(ZLoad);
  EXPECT_EQ(Known.Zero, APInt(32, 0xfffffffe));
  EXPECT_EQ(Known.One, APInt(32, 0));

  auto SLoad = DAG->getExtLoad(ISD::SEXTLOAD, Loc, Int32VT, DAG->getEntryNode(),
                               Ptr, Int8VT, MMO);
  Known = DAG->computeKnownBits(SLoad);
  EXPECT_EQ(Known.Zero, APInt(32, 0xfffffffe));
  EXPECT_EQ(Known.One, APInt(32, 0));
}

TEST_F(AArch64SelectionDAGTest, computeKnownBits_extload_knownnegative) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto Int64VT = EVT::getIntegerVT(Context, 64);
  auto Ptr = DAG->getConstant(0, Loc, Int64VT);
  auto PtrInfo =
      MachinePointerInfo::getFixedStack(DAG->getMachineFunction(), 0);
  AAMDNodes AA;
  MDBuilder MDHelper(*DAG->getContext());
  MDNode *Range = MDHelper.createRange(APInt(8, 0xf0), APInt(8, 0xff));
  MachineMemOperand *MMO = DAG->getMachineFunction().getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOLoad, 8, Align(8), AA, Range);

  auto ALoad = DAG->getExtLoad(ISD::EXTLOAD, Loc, Int32VT, DAG->getEntryNode(),
                               Ptr, Int8VT, MMO);
  KnownBits Known = DAG->computeKnownBits(ALoad);
  EXPECT_EQ(Known.Zero, APInt(32, 0));
  EXPECT_EQ(Known.One, APInt(32, 0xf0));

  auto ZLoad = DAG->getExtLoad(ISD::ZEXTLOAD, Loc, Int32VT, DAG->getEntryNode(),
                               Ptr, Int8VT, MMO);
  Known = DAG->computeKnownBits(ZLoad);
  EXPECT_EQ(Known.Zero, APInt(32, 0xffffff00));
  EXPECT_EQ(Known.One, APInt(32, 0x000000f0));

  auto SLoad = DAG->getExtLoad(ISD::SEXTLOAD, Loc, Int32VT, DAG->getEntryNode(),
                               Ptr, Int8VT, MMO);
  Known = DAG->computeKnownBits(SLoad);
  EXPECT_EQ(Known.Zero, APInt(32, 0));
  EXPECT_EQ(Known.One, APInt(32, 0xfffffff0));
}

TEST_F(AArch64SelectionDAGTest,
       computeKnownBits_AVGFLOORU_AVGFLOORS_AVGCEILU_AVGCEILS) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto Int16VT = EVT::getIntegerVT(Context, 16);
  auto Int8Vec8VT = EVT::getVectorVT(Context, Int8VT, 8);
  auto Int16Vec8VT = EVT::getVectorVT(Context, Int16VT, 8);

  SDValue UnknownOp0 = DAG->getRegister(0, Int8Vec8VT);
  SDValue UnknownOp1 = DAG->getRegister(1, Int8Vec8VT);

  SDValue ZextOp0 =
      DAG->getNode(ISD::ZERO_EXTEND, Loc, Int16Vec8VT, UnknownOp0);
  SDValue ZextOp1 =
      DAG->getNode(ISD::ZERO_EXTEND, Loc, Int16Vec8VT, UnknownOp1);
  // ZextOp0 = 00000000????????
  // ZextOp1 = 00000000????????
  // => (for all AVG* instructions)
  // Known.Zero = 1111111100000000 (0xFF00)
  // Known.One  = 0000000000000000 (0x0000)
  auto Zeroes = APInt(16, 0xFF00);
  auto Ones = APInt(16, 0x0000);

  SDValue AVGFLOORU =
      DAG->getNode(ISD::AVGFLOORU, Loc, Int16Vec8VT, ZextOp0, ZextOp1);
  KnownBits KnownAVGFLOORU = DAG->computeKnownBits(AVGFLOORU);
  EXPECT_EQ(KnownAVGFLOORU.Zero, Zeroes);
  EXPECT_EQ(KnownAVGFLOORU.One, Ones);

  SDValue AVGFLOORS =
      DAG->getNode(ISD::AVGFLOORS, Loc, Int16Vec8VT, ZextOp0, ZextOp1);
  KnownBits KnownAVGFLOORS = DAG->computeKnownBits(AVGFLOORS);
  EXPECT_EQ(KnownAVGFLOORS.Zero, Zeroes);
  EXPECT_EQ(KnownAVGFLOORS.One, Ones);

  SDValue AVGCEILU =
      DAG->getNode(ISD::AVGCEILU, Loc, Int16Vec8VT, ZextOp0, ZextOp1);
  KnownBits KnownAVGCEILU = DAG->computeKnownBits(AVGCEILU);
  EXPECT_EQ(KnownAVGCEILU.Zero, Zeroes);
  EXPECT_EQ(KnownAVGCEILU.One, Ones);

  SDValue AVGCEILS =
      DAG->getNode(ISD::AVGCEILS, Loc, Int16Vec8VT, ZextOp0, ZextOp1);
  KnownBits KnownAVGCEILS = DAG->computeKnownBits(AVGCEILS);
  EXPECT_EQ(KnownAVGCEILS.Zero, Zeroes);
  EXPECT_EQ(KnownAVGCEILS.One, Ones);
}

} // end namespace llvm
