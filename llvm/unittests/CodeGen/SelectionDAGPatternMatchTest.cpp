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
#include "llvm/IR/Module.h"
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
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        "riscv64", "", "+m,+f,+d,+v", Options, std::nullopt, std::nullopt,
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

TEST_F(SelectionDAGPatternMatchTest, matchVecShuffle) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto VInt32VT = EVT::getVectorVT(Context, Int32VT, 4);
  const std::array<int, 4> MaskData = {2, 0, 3, 1};
  const std::array<int, 4> OtherMaskData = {1, 2, 3, 4};
  ArrayRef<int> Mask;

  SDValue V0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, VInt32VT);
  SDValue V1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, VInt32VT);
  SDValue VecShuffleWithMask =
      DAG->getVectorShuffle(VInt32VT, DL, V0, V1, MaskData);

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(VecShuffleWithMask, m_Shuffle(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(VecShuffleWithMask,
                       m_Shuffle(m_Value(), m_Value(), m_Mask(Mask))));
  EXPECT_TRUE(
      sd_match(VecShuffleWithMask,
               m_Shuffle(m_Value(), m_Value(), m_SpecificMask(MaskData))));
  EXPECT_FALSE(
      sd_match(VecShuffleWithMask,
               m_Shuffle(m_Value(), m_Value(), m_SpecificMask(OtherMaskData))));
  EXPECT_TRUE(
      std::equal(MaskData.begin(), MaskData.end(), Mask.begin(), Mask.end()));
}

TEST_F(SelectionDAGPatternMatchTest, matchTernaryOp) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);

  SDValue Op0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Op1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, Int32VT);
  SDValue Op3 = DAG->getConstant(1, DL, Int32VT);

  SDValue ICMP_UGT = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETUGT);
  SDValue ICMP_EQ01 = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETEQ);
  SDValue ICMP_EQ10 = DAG->getSetCC(DL, MVT::i1, Op1, Op0, ISD::SETEQ);

  auto Int1VT = EVT::getIntegerVT(Context, 1);
  SDValue Cond = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 3, Int1VT);
  SDValue T = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 4, Int1VT);
  SDValue F = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 5, Int1VT);
  SDValue Select = DAG->getSelect(DL, MVT::i1, Cond, T, F);

  auto VInt32VT = EVT::getVectorVT(Context, Int32VT, 4);
  auto SmallVInt32VT = EVT::getVectorVT(Context, Int32VT, 2);
  auto Idx0 = DAG->getVectorIdxConstant(0, DL);
  auto Idx3 = DAG->getVectorIdxConstant(3, DL);
  SDValue V1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 6, VInt32VT);
  SDValue V2 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 7, VInt32VT);
  SDValue V3 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 8, SmallVInt32VT);
  SDValue VSelect = DAG->getNode(ISD::VSELECT, DL, VInt32VT, Cond, V1, V2);
  SDValue InsertSubvector =
      DAG->getNode(ISD::INSERT_SUBVECTOR, DL, VInt32VT, V2, V3, Idx0);

  SDValue ExtractELT =
      DAG->getNode(ISD::EXTRACT_VECTOR_ELT, DL, Int32VT, V1, Op3);

  using namespace SDPatternMatch;
  ISD::CondCode CC;
  EXPECT_TRUE(sd_match(ICMP_UGT, m_SetCC(m_Value(), m_Value(),
                                         m_SpecificCondCode(ISD::SETUGT))));
  EXPECT_TRUE(
      sd_match(ICMP_UGT, m_SetCC(m_Value(), m_Value(), m_CondCode(CC))));
  EXPECT_TRUE(CC == ISD::SETUGT);
  EXPECT_FALSE(sd_match(
      ICMP_UGT, m_SetCC(m_Value(), m_Value(), m_SpecificCondCode(ISD::SETLE))));

  EXPECT_TRUE(sd_match(ICMP_EQ01, m_SetCC(m_Specific(Op0), m_Specific(Op1),
                                          m_SpecificCondCode(ISD::SETEQ))));
  EXPECT_TRUE(sd_match(ICMP_EQ10, m_SetCC(m_Specific(Op1), m_Specific(Op0),
                                          m_SpecificCondCode(ISD::SETEQ))));
  EXPECT_FALSE(sd_match(ICMP_EQ01, m_SetCC(m_Specific(Op1), m_Specific(Op0),
                                           m_SpecificCondCode(ISD::SETEQ))));
  EXPECT_FALSE(sd_match(ICMP_EQ10, m_SetCC(m_Specific(Op0), m_Specific(Op1),
                                           m_SpecificCondCode(ISD::SETEQ))));
  EXPECT_TRUE(sd_match(ICMP_EQ01, m_c_SetCC(m_Specific(Op1), m_Specific(Op0),
                                            m_SpecificCondCode(ISD::SETEQ))));
  EXPECT_TRUE(sd_match(ICMP_EQ10, m_c_SetCC(m_Specific(Op0), m_Specific(Op1),
                                            m_SpecificCondCode(ISD::SETEQ))));

  EXPECT_TRUE(sd_match(
      Select, m_Select(m_Specific(Cond), m_Specific(T), m_Specific(F))));
  EXPECT_FALSE(sd_match(
      Select, m_Select(m_Specific(Cond), m_Specific(F), m_Specific(T))));
  EXPECT_FALSE(sd_match(ICMP_EQ01, m_Select(m_Specific(Op0), m_Specific(Op1),
                                            m_SpecificCondCode(ISD::SETEQ))));
  EXPECT_TRUE(sd_match(
      VSelect, m_VSelect(m_Specific(Cond), m_Specific(V1), m_Specific(V2))));
  EXPECT_FALSE(sd_match(
      Select, m_VSelect(m_Specific(Cond), m_Specific(V1), m_Specific(V2))));

  EXPECT_TRUE(sd_match(ExtractELT, m_ExtractElt(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(ExtractELT, m_ExtractElt(m_Value(), m_ConstInt())));
  EXPECT_TRUE(sd_match(ExtractELT, m_ExtractElt(m_Value(), m_SpecificInt(1))));

  EXPECT_TRUE(sd_match(InsertSubvector,
                       m_InsertSubvector(m_Value(), m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(
      InsertSubvector,
      m_InsertSubvector(m_Specific(V2), m_Specific(V3), m_Specific(Idx0))));
  EXPECT_TRUE(sd_match(
      InsertSubvector,
      m_InsertSubvector(m_Specific(V2), m_Specific(V3), m_SpecificInt(0))));
  EXPECT_FALSE(sd_match(
      InsertSubvector,
      m_InsertSubvector(m_Specific(V2), m_Specific(V3), m_Specific(Idx3))));
  EXPECT_FALSE(sd_match(
      InsertSubvector,
      m_InsertSubvector(m_Specific(V2), m_Specific(V3), m_SpecificInt(3))));
}

TEST_F(SelectionDAGPatternMatchTest, matchBinaryOp) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto Float32VT = EVT::getFloatingPointVT(32);
  auto BigVInt32VT = EVT::getVectorVT(Context, Int32VT, 8);
  auto VInt32VT = EVT::getVectorVT(Context, Int32VT, 4);

  SDValue V1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 6, VInt32VT);
  auto Idx0 = DAG->getVectorIdxConstant(0, DL);
  auto Idx1 = DAG->getVectorIdxConstant(1, DL);

  SDValue Op0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Op1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, Int32VT);
  SDValue Op2 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 3, Float32VT);
  SDValue Op3 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 8, Int32VT);
  SDValue Op4 = DAG->getConstant(1, DL, Int32VT);

  SDValue Add = DAG->getNode(ISD::ADD, DL, Int32VT, Op0, Op1);
  SDValue Sub = DAG->getNode(ISD::SUB, DL, Int32VT, Add, Op0);
  SDValue Mul = DAG->getNode(ISD::MUL, DL, Int32VT, Add, Sub);
  SDValue And = DAG->getNode(ISD::AND, DL, Int32VT, Op0, Op1);
  SDValue Xor = DAG->getNode(ISD::XOR, DL, Int32VT, Op1, Op0);
  SDValue Or  = DAG->getNode(ISD::OR, DL, Int32VT, Op0, Op1);
  SDValue DisOr =
      DAG->getNode(ISD::OR, DL, Int32VT, Op0, Op3, SDNodeFlags::Disjoint);
  SDValue SMax = DAG->getNode(ISD::SMAX, DL, Int32VT, Op0, Op1);
  SDValue SMin = DAG->getNode(ISD::SMIN, DL, Int32VT, Op1, Op0);
  SDValue UMax = DAG->getNode(ISD::UMAX, DL, Int32VT, Op0, Op1);
  SDValue UMin = DAG->getNode(ISD::UMIN, DL, Int32VT, Op1, Op0);
  SDValue Rotl = DAG->getNode(ISD::ROTL, DL, Int32VT, Op0, Op1);
  SDValue Rotr = DAG->getNode(ISD::ROTR, DL, Int32VT, Op1, Op0);

  SDValue ICMP_GT = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETGT);
  SDValue ICMP_GE = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETGE);
  SDValue ICMP_UGT = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETUGT);
  SDValue ICMP_UGE = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETUGE);
  SDValue ICMP_LT = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETLT);
  SDValue ICMP_LE = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETLE);
  SDValue ICMP_ULT = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETULT);
  SDValue ICMP_ULE = DAG->getSetCC(DL, MVT::i1, Op0, Op1, ISD::SETULE);
  SDValue SMaxLikeGT = DAG->getSelect(DL, MVT::i32, ICMP_GT, Op0, Op1);
  SDValue SMaxLikeGE = DAG->getSelect(DL, MVT::i32, ICMP_GE, Op0, Op1);
  SDValue UMaxLikeUGT = DAG->getSelect(DL, MVT::i32, ICMP_UGT, Op0, Op1);
  SDValue UMaxLikeUGE = DAG->getSelect(DL, MVT::i32, ICMP_UGE, Op0, Op1);
  SDValue SMinLikeLT = DAG->getSelect(DL, MVT::i32, ICMP_LT, Op0, Op1);
  SDValue SMinLikeLE = DAG->getSelect(DL, MVT::i32, ICMP_LE, Op0, Op1);
  SDValue UMinLikeULT = DAG->getSelect(DL, MVT::i32, ICMP_ULT, Op0, Op1);
  SDValue UMinLikeULE = DAG->getSelect(DL, MVT::i32, ICMP_ULE, Op0, Op1);

  SDValue SFAdd = DAG->getNode(ISD::STRICT_FADD, DL, {Float32VT, MVT::Other},
                               {DAG->getEntryNode(), Op2, Op2});

  SDValue Vec = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 9, BigVInt32VT);
  SDValue SubVec =
      DAG->getNode(ISD::EXTRACT_SUBVECTOR, DL, VInt32VT, Vec, Idx0);

  SDValue InsertELT =
      DAG->getNode(ISD::INSERT_VECTOR_ELT, DL, VInt32VT, V1, Op0, Op4);

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(Sub, m_BinOp(ISD::SUB, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Sub, m_Sub(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Add, m_c_BinOp(ISD::ADD, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Add, m_Add(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Add, m_AddLike(m_Value(), m_Value())));
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
  EXPECT_FALSE(sd_match(Or, m_DisjointOr(m_Value(), m_Value())));

  EXPECT_TRUE(sd_match(DisOr, m_Or(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(DisOr, m_DisjointOr(m_Value(), m_Value())));
  EXPECT_FALSE(sd_match(DisOr, m_Add(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(DisOr, m_AddLike(m_Value(), m_Value())));

  EXPECT_TRUE(sd_match(Rotl, m_Rotl(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(Rotr, m_Rotr(m_Value(), m_Value())));
  EXPECT_FALSE(sd_match(Rotl, m_Rotr(m_Value(), m_Value())));
  EXPECT_FALSE(sd_match(Rotr, m_Rotl(m_Value(), m_Value())));

  EXPECT_TRUE(sd_match(SMax, m_c_BinOp(ISD::SMAX, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMax, m_SMax(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMax, m_SMaxLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMaxLikeGT, m_SMaxLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMaxLikeGE, m_SMaxLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMin, m_c_BinOp(ISD::SMIN, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMin, m_SMin(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMin, m_SMinLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMinLikeLT, m_SMinLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(SMinLikeLE, m_SMinLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMax, m_c_BinOp(ISD::UMAX, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMax, m_UMax(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMax, m_UMaxLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMaxLikeUGT, m_UMaxLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMaxLikeUGE, m_UMaxLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMin, m_c_BinOp(ISD::UMIN, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMin, m_UMin(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMin, m_UMinLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMinLikeULT, m_UMinLike(m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(UMinLikeULE, m_UMinLike(m_Value(), m_Value())));

  SDValue BindVal;
  EXPECT_TRUE(sd_match(SFAdd, m_ChainedBinOp(ISD::STRICT_FADD, m_Value(BindVal),
                                             m_Deferred(BindVal))));
  EXPECT_FALSE(sd_match(SFAdd, m_ChainedBinOp(ISD::STRICT_FADD, m_OtherVT(),
                                              m_SpecificVT(Float32VT))));

  EXPECT_TRUE(sd_match(SubVec, m_ExtractSubvector(m_Value(), m_Value())));
  EXPECT_TRUE(
      sd_match(SubVec, m_ExtractSubvector(m_Specific(Vec), m_Specific(Idx0))));
  EXPECT_TRUE(
      sd_match(SubVec, m_ExtractSubvector(m_Specific(Vec), m_SpecificInt(0))));
  EXPECT_FALSE(
      sd_match(SubVec, m_ExtractSubvector(m_Specific(Vec), m_Specific(Idx1))));
  EXPECT_FALSE(
      sd_match(SubVec, m_ExtractSubvector(m_Specific(Vec), m_SpecificInt(1))));

  EXPECT_TRUE(
      sd_match(InsertELT, m_InsertElt(m_Value(), m_Value(), m_Value())));
  EXPECT_TRUE(
      sd_match(InsertELT, m_InsertElt(m_Value(), m_Value(), m_ConstInt())));
  EXPECT_TRUE(
      sd_match(InsertELT, m_InsertElt(m_Value(), m_Value(), m_SpecificInt(1))));
}

TEST_F(SelectionDAGPatternMatchTest, matchUnaryOp) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto Int64VT = EVT::getIntegerVT(Context, 64);
  auto FloatVT = EVT::getFloatingPointVT(32);

  SDValue Op0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Op1 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int64VT);
  SDValue Op2 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, FloatVT);  
  SDValue Op3 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 3, Int32VT);

  SDValue ZExt = DAG->getNode(ISD::ZERO_EXTEND, DL, Int64VT, Op0);
  SDValue ZExtNNeg =
      DAG->getNode(ISD::ZERO_EXTEND, DL, Int64VT, Op3, SDNodeFlags::NonNeg);
  SDValue SExt = DAG->getNode(ISD::SIGN_EXTEND, DL, Int64VT, Op0);
  SDValue Trunc = DAG->getNode(ISD::TRUNCATE, DL, Int32VT, Op1);

  SDValue Sub = DAG->getNode(ISD::SUB, DL, Int32VT, Trunc, Op0);
  SDValue Neg = DAG->getNegative(Op0, DL, Int32VT);
  SDValue Not = DAG->getNOT(DL, Op0, Int32VT);

  SDValue VScale = DAG->getVScale(DL, Int32VT, APInt::getMaxValue(32));

  SDValue FPToSI = DAG->getNode(ISD::FP_TO_SINT, DL, FloatVT, Op2);
  SDValue FPToUI = DAG->getNode(ISD::FP_TO_UINT, DL, FloatVT, Op2);

  SDValue Bcast = DAG->getNode(ISD::BITCAST, DL, FloatVT, Op0);
  SDValue Brev = DAG->getNode(ISD::BITREVERSE, DL, Int32VT, Op0);
  SDValue Bswap = DAG->getNode(ISD::BSWAP, DL, Int32VT, Op0);

  SDValue Ctpop = DAG->getNode(ISD::CTPOP, DL, Int32VT, Op0);
  SDValue Ctlz = DAG->getNode(ISD::CTLZ, DL, Int32VT, Op0);
  SDValue Cttz = DAG->getNode(ISD::CTTZ, DL, Int32VT, Op0);

  using namespace SDPatternMatch;
  EXPECT_TRUE(sd_match(ZExt, m_UnaryOp(ISD::ZERO_EXTEND, m_Value())));
  EXPECT_TRUE(sd_match(SExt, m_SExt(m_Value())));
  EXPECT_TRUE(sd_match(SExt, m_SExtLike(m_Value())));
  ASSERT_TRUE(ZExtNNeg->getFlags().hasNonNeg());
  EXPECT_FALSE(sd_match(ZExtNNeg, m_SExt(m_Value())));
  EXPECT_TRUE(sd_match(ZExtNNeg, m_NNegZExt(m_Value())));
  EXPECT_FALSE(sd_match(ZExt, m_NNegZExt(m_Value())));
  EXPECT_TRUE(sd_match(ZExtNNeg, m_SExtLike(m_Value())));
  EXPECT_FALSE(sd_match(ZExt, m_SExtLike(m_Value())));
  EXPECT_TRUE(sd_match(Trunc, m_Trunc(m_Specific(Op1))));

  EXPECT_TRUE(sd_match(Neg, m_Neg(m_Value())));
  EXPECT_TRUE(sd_match(Not, m_Not(m_Value())));
  EXPECT_FALSE(sd_match(ZExt, m_Neg(m_Value())));
  EXPECT_FALSE(sd_match(Sub, m_Neg(m_Value())));
  EXPECT_FALSE(sd_match(Neg, m_Not(m_Value())));
  EXPECT_TRUE(sd_match(VScale, m_VScale(m_Value())));

  EXPECT_TRUE(sd_match(FPToUI, m_FPToUI(m_Value())));
  EXPECT_TRUE(sd_match(FPToSI, m_FPToSI(m_Value())));
  EXPECT_FALSE(sd_match(FPToUI, m_FPToSI(m_Value())));
  EXPECT_FALSE(sd_match(FPToSI, m_FPToUI(m_Value())));

  EXPECT_TRUE(sd_match(Bcast, m_BitCast(m_Value())));
  EXPECT_TRUE(sd_match(Bcast, m_BitCast(m_SpecificVT(MVT::i32))));
  EXPECT_TRUE(sd_match(Brev, m_BitReverse(m_Value())));
  EXPECT_TRUE(sd_match(Bswap, m_BSwap(m_Value())));
  EXPECT_FALSE(sd_match(Bcast, m_BitReverse(m_Value())));
  EXPECT_FALSE(sd_match(Bcast, m_BitCast(m_SpecificVT(MVT::f32))));
  EXPECT_FALSE(sd_match(Brev, m_BSwap(m_Value())));
  EXPECT_FALSE(sd_match(Bswap, m_BitReverse(m_Value())));

  EXPECT_TRUE(sd_match(Ctpop, m_Ctpop(m_Value())));
  EXPECT_TRUE(sd_match(Ctlz, m_Ctlz(m_Value())));
  EXPECT_TRUE(sd_match(Cttz, m_Cttz(m_Value())));
  EXPECT_FALSE(sd_match(Ctpop, m_Ctlz(m_Value())));
  EXPECT_FALSE(sd_match(Ctlz, m_Cttz(m_Value())));
  EXPECT_FALSE(sd_match(Cttz, m_Ctlz(m_Value())));
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
  SDValue SetCC = DAG->getSetCC(DL, Int32VT, Arg0, Const3, ISD::SETULT);

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

  ISD::CondCode CC;
  EXPECT_TRUE(sd_match(
      SetCC, m_Node(ISD::SETCC, m_Value(), m_Value(), m_CondCode(CC))));
  EXPECT_EQ(CC, ISD::SETULT);
  EXPECT_TRUE(sd_match(SetCC, m_Node(ISD::SETCC, m_Value(), m_Value(),
                                     m_SpecificCondCode(ISD::SETULT))));

  SDValue UndefInt32VT = DAG->getUNDEF(Int32VT);
  SDValue UndefVInt32VT = DAG->getUNDEF(VInt32VT);
  EXPECT_TRUE(sd_match(UndefInt32VT, m_Undef()));
  EXPECT_TRUE(sd_match(UndefVInt32VT, m_Undef()));
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
  EXPECT_TRUE(sd_match(Add, m_NoneOf(m_Opc(ISD::SUB), m_Opc(ISD::MUL))));
}

TEST_F(SelectionDAGPatternMatchTest, optionalResizing) {
  SDLoc DL;
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto Int64VT = EVT::getIntegerVT(Context, 64);

  SDValue Op32 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int32VT);
  SDValue Op64 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, Int64VT);
  SDValue ZExt = DAG->getNode(ISD::ZERO_EXTEND, DL, Int64VT, Op32);
  SDValue SExt = DAG->getNode(ISD::SIGN_EXTEND, DL, Int64VT, Op32);
  SDValue AExt = DAG->getNode(ISD::ANY_EXTEND, DL, Int64VT, Op32);
  SDValue Trunc = DAG->getNode(ISD::TRUNCATE, DL, Int32VT, Op64);

  using namespace SDPatternMatch;
  SDValue A;
  EXPECT_TRUE(sd_match(Op32, m_ZExtOrSelf(m_Value(A))));
  EXPECT_TRUE(A == Op32);
  EXPECT_TRUE(sd_match(ZExt, m_ZExtOrSelf(m_Value(A))));
  EXPECT_TRUE(A == Op32);
  EXPECT_TRUE(sd_match(Op64, m_SExtOrSelf(m_Value(A))));
  EXPECT_TRUE(A == Op64);
  EXPECT_TRUE(sd_match(SExt, m_SExtOrSelf(m_Value(A))));
  EXPECT_TRUE(A == Op32);
  EXPECT_TRUE(sd_match(Op32, m_AExtOrSelf(m_Value(A))));
  EXPECT_TRUE(A == Op32);
  EXPECT_TRUE(sd_match(AExt, m_AExtOrSelf(m_Value(A))));
  EXPECT_TRUE(A == Op32);
  EXPECT_TRUE(sd_match(Op64, m_TruncOrSelf(m_Value(A))));
  EXPECT_TRUE(A == Op64);
  EXPECT_TRUE(sd_match(Trunc, m_TruncOrSelf(m_Value(A))));
  EXPECT_TRUE(A == Op64);
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

  unsigned getNumOperands(SDValue N) const {
    return N->isVPOpcode() ? N->getNumOperands() - 2 : N->getNumOperands();
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
  SDValue Add = DAG->getNode(ISD::ADD, DL, VInt32VT, {Vector0, Vector0});

  using namespace SDPatternMatch;
  VPMatchContext VPCtx(DAG.get());
  EXPECT_TRUE(sd_context_match(VPAdd, VPCtx, m_Opc(ISD::ADD)));
  EXPECT_TRUE(
      sd_context_match(VPAdd, VPCtx, m_Node(ISD::ADD, m_Value(), m_Value())));
  // VPMatchContext can't match pattern using explicit VP Opcode
  EXPECT_FALSE(sd_context_match(VPAdd, VPCtx,
                                m_Node(ISD::VP_ADD, m_Value(), m_Value())));
  EXPECT_FALSE(sd_context_match(
      VPAdd, VPCtx,
      m_Node(ISD::VP_ADD, m_Value(), m_Value(), m_Value(), m_Value())));
  // Check Binary Op Pattern
  EXPECT_TRUE(sd_context_match(VPAdd, VPCtx, m_Add(m_Value(), m_Value())));
  // VP_REDUCE_ADD doesn't have a based opcode, so we use a normal
  // sd_match before switching to VPMatchContext when checking VPAdd.
  EXPECT_TRUE(sd_match(VPReduceAdd, m_Node(ISD::VP_REDUCE_ADD, m_Value(),
                                           m_Context(VPCtx, m_Opc(ISD::ADD)),
                                           m_Value(), m_Value())));
  // non-vector predicated should match too
  EXPECT_TRUE(sd_context_match(Add, VPCtx, m_Opc(ISD::ADD)));
  EXPECT_TRUE(
      sd_context_match(Add, VPCtx, m_Node(ISD::ADD, m_Value(), m_Value())));
  EXPECT_FALSE(sd_context_match(
      Add, VPCtx,
      m_Node(ISD::ADD, m_Value(), m_Value(), m_Value(), m_Value())));
  EXPECT_TRUE(sd_context_match(Add, VPCtx, m_Add(m_Value(), m_Value())));
}

TEST_F(SelectionDAGPatternMatchTest, matchVPWithBasicContext) {
  SDLoc DL;
  auto BoolVT = EVT::getIntegerVT(Context, 1);
  auto Int32VT = EVT::getIntegerVT(Context, 32);
  auto VInt32VT = EVT::getVectorVT(Context, Int32VT, 4);
  auto MaskVT = EVT::getVectorVT(Context, BoolVT, 4);

  SDValue Vector0 = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 1, VInt32VT);
  SDValue Mask = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 2, MaskVT);
  SDValue EL = DAG->getCopyFromReg(DAG->getEntryNode(), DL, 3, Int32VT);

  SDValue VPAdd =
      DAG->getNode(ISD::VP_ADD, DL, VInt32VT, Vector0, Vector0, Mask, EL);

  using namespace SDPatternMatch;
  EXPECT_FALSE(sd_match(VPAdd, m_Node(ISD::VP_ADD, m_Value(), m_Value())));
  EXPECT_TRUE(sd_match(
      VPAdd, m_Node(ISD::VP_ADD, m_Value(), m_Value(), m_Value(), m_Value())));
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
