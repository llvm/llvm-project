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

using namespace llvm;

namespace {

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
    TM = std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine*>(
        T->createTargetMachine("AArch64", "", "", Options, None, None,
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

} // end anonymous namespace
