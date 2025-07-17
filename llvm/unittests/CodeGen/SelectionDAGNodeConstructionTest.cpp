//===---- llvm/unittest/CodeGen/SelectionDAGPatternMatchTest.cpp  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SDPatternMatch.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

class SelectionDAGnodeConstructionTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    StringRef Assembly = "@g = global i32 0\n"
                         "@g_alias = alias i32, i32* @g\n"
                         "define i32 @f() {\n"
                         "  %1 = load i32, i32* @g\n"
                         "  ret i32 %1\n"
                         "}";

    Triple TargetTriple("riscv64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    // FIXME: These tests do not depend on RISCV specifically, but we have to
    // initialize a target. A skeleton Target for unittests would allow us to
    // always run these tests.
    if (!T)
      GTEST_SKIP();

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TargetTriple, "", "+m,+f,+d,+v", Options, std::nullopt, std::nullopt,
        CodeGenOptLevel::Aggressive));
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
    G = M->getGlobalVariable("g");
    if (!G)
      report_fatal_error("G?");
    AliasedG = M->getNamedAlias("g_alias");
    if (!AliasedG)
      report_fatal_error("AliasedG?");

    MachineModuleInfo MMI(TM.get());

    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           MMI.getContext(), 0);

    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOptLevel::None);
    if (!DAG)
      report_fatal_error("DAG?");
    OptimizationRemarkEmitter ORE(F);
    FunctionAnalysisManager FAM;
    FAM.registerPass([&] { return TM->getTargetIRAnalysis(); });

    TargetTransformInfo TTI = TM->getTargetIRAnalysis().run(*F, FAM);
    DAG->init(*MF, ORE, nullptr, nullptr, nullptr, nullptr, nullptr, MMI,
              nullptr, TTI.hasBranchDivergence(F));
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
  GlobalVariable *G;
  GlobalAlias *AliasedG;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

TEST_F(SelectionDAGnodeConstructionTest, ADD) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Undef, Op), Undef);
  EXPECT_EQ(DAG->getNode(ISD::ADD, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, AND) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::AND, DL, Int32VT, Undef, Undef), Zero);
}

TEST_F(SelectionDAGnodeConstructionTest, MUL) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::MUL, DL, Int32VT, Undef, Undef), Zero);
}

TEST_F(SelectionDAGnodeConstructionTest, OR) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Undef, Op), AllOnes);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::OR, DL, Int32VT, Undef, Undef), AllOnes);
}

TEST_F(SelectionDAGnodeConstructionTest, SADDSAT) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Undef, Op), AllOnes);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::SADDSAT, DL, Int32VT, Undef, Undef), AllOnes);
}

TEST_F(SelectionDAGnodeConstructionTest, SDIV) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Op, Poison), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Poison, Undef), Undef);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Undef, Poison), Undef);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SDIV, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, SMAX) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue MaxInt = DAG->getConstant(APInt::getSignedMaxValue(32), DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, Int32VT, Op, Undef), MaxInt);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, Int32VT, Undef, Op), MaxInt);
  EXPECT_EQ(DAG->getNode(ISD::SMAX, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, SMIN) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue MinInt = DAG->getConstant(APInt::getSignedMinValue(32), DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, Int32VT, Op, Undef), MinInt);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, Int32VT, Undef, Op), MinInt);
  EXPECT_EQ(DAG->getNode(ISD::SMIN, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, SREM) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Op, Poison), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Poison, Undef), Undef);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Undef, Poison), Undef);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::SREM, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, SSUBSAT) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::SSUBSAT, DL, Int32VT, Undef, Undef), Zero);
}

TEST_F(SelectionDAGnodeConstructionTest, SUB) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Undef, Op), Undef);
  EXPECT_EQ(DAG->getNode(ISD::SUB, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, UADDSAT) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Undef, Op), AllOnes);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::UADDSAT, DL, Int32VT, Undef, Undef), AllOnes);
}

TEST_F(SelectionDAGnodeConstructionTest, UDIV) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Op, Poison), Undef);
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Poison, Undef), Undef);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Undef, Poison), Undef);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UDIV, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, UMAX) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue AllOnes = DAG->getAllOnesConstant(DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, Int32VT, Op, Undef), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, Int32VT, Undef, Op), AllOnes);
  EXPECT_EQ(DAG->getNode(ISD::UMAX, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, UMIN) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, Int32VT, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, Int32VT, Undef, Op), Zero);
  EXPECT_EQ(DAG->getNode(ISD::UMIN, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, UREM) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Op, Poison), Undef);
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Poison, Undef), Undef);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Undef, Poison), Undef);

  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::UREM, DL, Int32VT, Undef, Undef), Undef);
}

TEST_F(SelectionDAGnodeConstructionTest, USUBSAT) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Poison, Op), Poison);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Poison, Undef), Poison);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Undef, Poison), Poison);

  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Op, Undef), Zero);
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Undef, Op), Zero);
  // TODO: Should be undef.
  EXPECT_EQ(DAG->getNode(ISD::USUBSAT, DL, Int32VT, Undef, Undef), Zero);
}

TEST_F(SelectionDAGnodeConstructionTest, XOR) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  SDValue Op = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Poison = DAG->getPOISON(Int32VT);
  SDValue Undef = DAG->getUNDEF(Int32VT);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);

  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Op, Poison), Poison);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Poison, Op), Poison);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Poison, Undef), Zero);
  // TODO: Should be poison.
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Undef, Poison), Zero);

  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Op, Undef), Undef);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Undef, Op), Undef);
  EXPECT_EQ(DAG->getNode(ISD::XOR, DL, Int32VT, Undef, Undef), Zero);
}
