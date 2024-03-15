//===---- llvm/unittest/CodeGen/SelectionDAGPatternMatchTest.cpp  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SDPatternMatch.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

class SelectionDAGPatternMatchTest : public testing::Test {
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
    TM = std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
        T->createTargetMachine("riscv64", "", "+m,+f,+d,+v", Options,
                               std::nullopt, std::nullopt,
                               CodeGenOptLevel::Aggressive)));
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
                                           0, MMI);

    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOptLevel::None);
    if (!DAG)
      report_fatal_error("DAG?");
    OptimizationRemarkEmitter ORE(F);
    DAG->init(*MF, ORE, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
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
  GlobalVariable *G;
  GlobalAlias *AliasedG;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

TEST_F(SelectionDAGPatternMatchTest, matchValueType) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto Float32VT = EVT::getFloatingPointVT(32);
  auto VInt32VT = EVT::getVectorVT(Context, Int32VT, 4);

  SDValue Op0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Op1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, Float32VT);
  SDValue Op2 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, VInt32VT);

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(Op0, m_SpecificVT(Int32VT)));
  EVT BindVT;
  EXPECT_TRUE(sd_match(Op1, m_VT(BindVT)));
  EXPECT_EQ(BindVT, Float32VT);
  EXPECT_TRUE(sd_match(Op0, m_IntegerVT()));
  EXPECT_TRUE(sd_match(Op1, m_FloatingPointVT()));
  EXPECT_TRUE(sd_match(Op2, m_VectorVT()));
  EXPECT_FALSE(sd_match(Op2, m_ScalableVectorVT()));
}

TEST_F(SelectionDAGPatternMatchTest, matchBinaryOp) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto Float32VT = EVT::getFloatingPointVT(32);

  SDValue Op0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Op1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, Int32VT);
  SDValue Op2 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 3, Float32VT);

  SDValue Add = DAG->getNode(ISD::ADD, DL, Int32VT, Op0, Op1);
  SDValue Sub = DAG->getNode(ISD::SUB, DL, Int32VT, Add, Op0);
  SDValue Mul = DAG->getNode(ISD::MUL, DL, Int32VT, Add, Sub);
  SDValue And = DAG->getNode(ISD::AND, DL, Int32VT, Op0, Op1);
  SDValue Xor = DAG->getNode(ISD::XOR, DL, Int32VT, Op1, Op0);
  SDValue Or  = DAG->getNode(ISD::OR, DL, Int32VT, Op0, Op1);
  SDValue SMax = DAG->getNode(ISD::SMAX, DL, Int32VT, Op0, Op1);
  SDValue SMin = DAG->getNode(ISD::SMIN, DL, Int32VT, Op1, Op0);
  SDValue UMax = DAG->getNode(ISD::UMAX, DL, Int32VT, Op0, Op1);
  SDValue UMin = DAG->getNode(ISD::UMIN, DL, Int32VT, Op1, Op0);

  SDValue SFAdd = DAG->getNode(ISD::STRICT_FADD, DL, {Float32VT, MVT::Other},
                               {DAG->getEntryNode(), Op2, Op2});

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(Sub, m_BinOp(ISD::SUB, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Sub, m_Sub(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Add, m_c_BinOp(ISD::ADD, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Add, m_Add(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(
      Mul, m_Mul(m_OneUse(m_Opc(ISD::SUB)), m_NUses<2>(m_Specific(Add)))));
  EXPECT_TRUE(
      sd_match(SFAdd, m_ChainedBinOp(ISD::STRICT_FADD, m_SpecificVT(Float32VT),
                                     m_SpecificVT(Float32VT))));

  EXPECT_TRUE(sd_match(And, m_c_BinOp(ISD::AND, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(And, m_And(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Xor, m_c_BinOp(ISD::XOR, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Xor, m_Xor(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Or, m_c_BinOp(ISD::OR, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Or, m_Or(m_Value(), m_Value())));

  EXPECT_TRUE(sd_match(SMax, m_c_BinOp(ISD::SMAX, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMax, m_SMax(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMin, m_c_BinOp(ISD::SMIN, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMin, m_SMin(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMax, m_c_BinOp(ISD::UMAX, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMax, m_UMax(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMin, m_c_BinOp(ISD::UMIN, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMin, m_UMin(m_Value(), m_Value())));

  SDValue BindVal;
  EXPECT_TRUE(sd_match(SFAdd, m_ChainedBinOp(ISD::STRICT_FADD, m_Value(BindVal),
                                             m_Deferred(BindVal))));
  EXPECT_FALSE(sd_match(SFAdd, m_ChainedBinOp(ISD::STRICT_FADD, m_OtherVT(),
                                              m_SpecificVT(Float32VT))));
}

TEST_F(SelectionDAGPatternMatchTest, matchUnaryOp) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto Int64VT = EVT::getIntegerVT(Context, 64);

  SDValue Op0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Op1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int64VT);

  SDValue ZExt = DAG->getNode(ISD::ZERO_EXTEND, DL, Int64VT, Op0);
  SDValue SExt = DAG->getNode(ISD::SIGN_EXTEND, DL, Int64VT, Op0);
  SDValue Trunc = DAG->getNode(ISD::TRUNCATE, DL, Int32VT, Op1);

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(ZExt, m_UnaryOp(ISD::ZERO_EXTEND, m_Value())));
  EXPECT_TRUE(sd_match(SExt, m_SExt(m_Value())));
  EXPECT_TRUE(sd_match(Trunc, m_Trunc(m_Specific(Op1))));
}

TEST_F(SelectionDAGPatternMatchTest, matchConstants) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto VInt32VT = EVT::getVectorVT(Context, Int32VT, 4);

  SDValue Arg0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);

  SDValue Const3 = DAG->getConstant(3, DL, Int32VT);
  SDValue Const87 = DAG->getConstant(87, DL, Int32VT);
  SDValue Splat = DAG->getSplat(VInt32VT, DL, Arg0);
  SDValue ConstSplat = DAG->getSplat(VInt32VT, DL, Const3);
  SDValue Zero = DAG->getConstant(0, DL, Int32VT);
  SDValue One = DAG->getConstant(1, DL, Int32VT);
  SDValue AllOnes = DAG->getConstant(APInt::getAllOnes(32), DL, Int32VT);

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(Const87, m_ConstInt()));
  EXPECT_FALSE(sd_match(Arg0, m_ConstInt()));
  APInt ConstVal;
  EXPECT_TRUE(sd_match(ConstSplat, m_ConstInt(ConstVal)));
  EXPECT_EQ(ConstVal, 3);
  EXPECT_FALSE(sd_match(Splat, m_ConstInt()));

  EXPECT_TRUE(sd_match(Const87, m_SpecificInt(87)));
  EXPECT_TRUE(sd_match(Const3, m_SpecificInt(ConstVal)));
  EXPECT_TRUE(sd_match(AllOnes, m_AllOnes()));

  EXPECT_TRUE(sd_match(Zero, DAG.get(), m_False()));
  EXPECT_TRUE(sd_match(One, DAG.get(), m_True()));
  EXPECT_FALSE(sd_match(AllOnes, DAG.get(), m_True()));
}

TEST_F(SelectionDAGPatternMatchTest, patternCombinators) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);

  SDValue Op0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Op1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, Int32VT);

  SDValue Add = DAG->getNode(ISD::ADD, DL, Int32VT, Op0, Op1);
  SDValue Sub = DAG->getNode(ISD::SUB, DL, Int32VT, Add, Op0);

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(
      Sub, m_AnyOf(m_Opc(ISD::ADD), m_Opc(ISD::SUB), m_Opc(ISD::MUL))));
  EXPECT_TRUE(sd_match(Add, m_AllOf(m_Opc(ISD::ADD), m_OneUse())));
}

TEST_F(SelectionDAGPatternMatchTest, matchNode) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);

  SDValue Op0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Op1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, Int32VT);

  SDValue Add = DAG->getNode(ISD::ADD, DL, Int32VT, Op0, Op1);

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(Add, m_Node(ISD::ADD, m_Value(), m_Value())));
  EXPECT_FALSE(sd_match(Add, m_Node(ISD::SUB, m_Value(), m_Value())));
  EXPECT_FALSE(sd_match(Add, m_Node(ISD::ADD, m_Value())));
  EXPECT_FALSE(
      sd_match(Add, m_Node(ISD::ADD, m_Value(), m_Value(), m_Value())));
  EXPECT_FALSE(sd_match(Add, m_Node(ISD::ADD, m_ConstInt(), m_Value())));
}

namespace {
struct VPMatchContext : public SDPatternMatch::BasicMatchContext {
  using SDPatternMatch::BasicMatchContext::BasicMatchContext;

  bool match(SDValue OpVal, unsigned Opc) const {
    if (!OpVal->isVPOpcode())
      return OpVal->getOpcode() == Opc;

    auto BaseOpc = ISD::getBaseOpcodeForVP(OpVal->getOpcode(), false);
    return BaseOpc.has_value() && *BaseOpc == Opc;
  }
};
} // anonymous namespace
TEST_F(SelectionDAGPatternMatchTest, matchContext) {
  SDLoc DL;
  auto BoolVT = EVT::getIntegerVT(Context, 1);
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto VInt32VT = EVT::getVectorVT(Context, Int32VT, 4);
  auto MaskVT = EVT::getVectorVT(Context, BoolVT, 4);

  SDValue Scalar0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Vector0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, VInt32VT);
  SDValue Mask0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 3, MaskVT);

  SDValue VPAdd = DAG->getNode(ISD::VP_ADD, DL, VInt32VT,
                               {Vector0, Vector0, Mask0, Scalar0});
  SDValue VPReduceAdd = DAG->getNode(ISD::VP_REDUCE_ADD, DL, Int32VT,
                                     {Scalar0, VPAdd, Mask0, Scalar0});

  using namespace SDPatternMatch;
  VPMatchContext VPCtx(DAG.get());
  EXPECT_TRUE(sd_context_match(VPAdd, VPCtx, m_Opc(ISD::ADD)));
  // VP_REDUCE_ADD doesn't have a based opcode, so we use a normal
  // sd_match before switching to VPMatchContext when checking VPAdd.
  EXPECT_TRUE(sd_match(VPReduceAdd, m_Node(ISD::VP_REDUCE_ADD, m_Value(),
                                           m_Context(VPCtx, m_Opc(ISD::ADD)),
                                           m_Value(), m_Value())));
}

TEST_F(SelectionDAGPatternMatchTest, matchAdvancedProperties) {
  SDLoc DL;
  auto Int16VT = EVT::getIntegerVT(Context, 16);
  auto Int64VT = EVT::getIntegerVT(Context, 64);

  SDValue Op0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int64VT);
  SDValue Op1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, Int16VT);

  SDValue Add = DAG->getNode(ISD::ADD, DL, Int64VT, Op0, Op0);

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(Op0, DAG.get(), m_LegalType(m_Value())));
  EXPECT_FALSE(sd_match(Op1, DAG.get(), m_LegalType(m_Value())));
  EXPECT_TRUE(sd_match(Add, DAG.get(),
                       m_LegalOp(m_IntegerVT(m_Add(m_Value(), m_Value())))));
}
