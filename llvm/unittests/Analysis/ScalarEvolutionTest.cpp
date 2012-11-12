//===- ScalarEvolutionsTest.cpp - ScalarEvolution unit tests --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

// We use this fixture to ensure that we clean up ScalarEvolution before
// deleting the PassManager.
class ScalarEvolutionsTest : public testing::Test {
protected:
  ScalarEvolutionsTest() : M("", Context), SE(*new ScalarEvolution) {}
  ~ScalarEvolutionsTest() {
    // Manually clean up, since we allocated new SCEV objects after the
    // pass was finished.
    SE.releaseMemory();
  }
  LLVMContext Context;
  Module M;
  PassManager PM;
  ScalarEvolution &SE;
};

TEST_F(ScalarEvolutionsTest, SCEVUnknownRAUW) {
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context),
                                              std::vector<Type *>(), false);
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));
  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
  ReturnInst::Create(Context, 0, BB);

  Type *Ty = Type::getInt1Ty(Context);
  Constant *Init = Constant::getNullValue(Ty);
  Value *V0 = new GlobalVariable(M, Ty, false, GlobalValue::ExternalLinkage, Init, "V0");
  Value *V1 = new GlobalVariable(M, Ty, false, GlobalValue::ExternalLinkage, Init, "V1");
  Value *V2 = new GlobalVariable(M, Ty, false, GlobalValue::ExternalLinkage, Init, "V2");

  // Create a ScalarEvolution and "run" it so that it gets initialized.
  PM.add(&SE);
  PM.run(M);

  const SCEV *S0 = SE.getSCEV(V0);
  const SCEV *S1 = SE.getSCEV(V1);
  const SCEV *S2 = SE.getSCEV(V2);

  const SCEV *P0 = SE.getAddExpr(S0, S0);
  const SCEV *P1 = SE.getAddExpr(S1, S1);
  const SCEV *P2 = SE.getAddExpr(S2, S2);

  const SCEVMulExpr *M0 = cast<SCEVMulExpr>(P0);
  const SCEVMulExpr *M1 = cast<SCEVMulExpr>(P1);
  const SCEVMulExpr *M2 = cast<SCEVMulExpr>(P2);

  EXPECT_EQ(cast<SCEVConstant>(M0->getOperand(0))->getValue()->getZExtValue(),
            2u);
  EXPECT_EQ(cast<SCEVConstant>(M1->getOperand(0))->getValue()->getZExtValue(),
            2u);
  EXPECT_EQ(cast<SCEVConstant>(M2->getOperand(0))->getValue()->getZExtValue(),
            2u);

  // Before the RAUWs, these are all pointing to separate values.
  EXPECT_EQ(cast<SCEVUnknown>(M0->getOperand(1))->getValue(), V0);
  EXPECT_EQ(cast<SCEVUnknown>(M1->getOperand(1))->getValue(), V1);
  EXPECT_EQ(cast<SCEVUnknown>(M2->getOperand(1))->getValue(), V2);

  // Do some RAUWs.
  V2->replaceAllUsesWith(V1);
  V1->replaceAllUsesWith(V0);

  // After the RAUWs, these should all be pointing to V0.
  EXPECT_EQ(cast<SCEVUnknown>(M0->getOperand(1))->getValue(), V0);
  EXPECT_EQ(cast<SCEVUnknown>(M1->getOperand(1))->getValue(), V0);
  EXPECT_EQ(cast<SCEVUnknown>(M2->getOperand(1))->getValue(), V0);
}

TEST_F(ScalarEvolutionsTest, SCEVMultiplyAddRecs) {
  Type *Ty = Type::getInt32Ty(Context);
  SmallVector<Type *, 10> Types;
  Types.append(10, Ty);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context), Types, false);
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));
  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
  ReturnInst::Create(Context, 0, BB);

  // Create a ScalarEvolution and "run" it so that it gets initialized.
  PM.add(&SE);
  PM.run(M);

  // It's possible to produce an empty loop through the default constructor,
  // but you can't add any blocks to it without a LoopInfo pass.
  Loop L;
  const_cast<std::vector<BasicBlock*>&>(L.getBlocks()).push_back(BB);

  Function::arg_iterator AI = F->arg_begin();
  SmallVector<const SCEV *, 5> A;
  A.push_back(SE.getSCEV(&*AI++));
  A.push_back(SE.getSCEV(&*AI++));
  A.push_back(SE.getSCEV(&*AI++));
  A.push_back(SE.getSCEV(&*AI++));
  A.push_back(SE.getSCEV(&*AI++));
  const SCEV *A_rec = SE.getAddRecExpr(A, &L, SCEV::FlagAnyWrap);

  SmallVector<const SCEV *, 5> B;
  B.push_back(SE.getSCEV(&*AI++));
  B.push_back(SE.getSCEV(&*AI++));
  B.push_back(SE.getSCEV(&*AI++));
  B.push_back(SE.getSCEV(&*AI++));
  B.push_back(SE.getSCEV(&*AI++));
  const SCEV *B_rec = SE.getAddRecExpr(B, &L, SCEV::FlagAnyWrap);

  /* Spot check that we perform this transformation:
     {A0,+,A1,+,A2,+,A3,+,A4} * {B0,+,B1,+,B2,+,B3,+,B4} =
     {A0*B0,+,
      A1*B0 + A0*B1 + A1*B1,+,
      A2*B0 + 2A1*B1 + A0*B2 + 2A2*B1 + 2A1*B2 + A2*B2,+,
      A3*B0 + 3A2*B1 + 3A1*B2 + A0*B3 + 3A3*B1 + 6A2*B2 + 3A1*B3 + 3A3*B2 +
        3A2*B3 + A3*B3,+,
      A4*B0 + 4A3*B1 + 6A2*B2 + 4A1*B3 + A0*B4 + 4A4*B1 + 12A3*B2 + 12A2*B3 +
        4A1*B4 + 6A4*B2 + 12A3*B3 + 6A2*B4 + 4A4*B3 + 4A3*B4 + A4*B4,+,
      5A4*B1 + 10A3*B2 + 10A2*B3 + 5A1*B4 + 20A4*B2 + 30A3*B3 + 20A2*B4 +
        30A4*B3 + 30A3*B4 + 20A4*B4,+,
      15A4*B2 + 20A3*B3 + 15A2*B4 + 60A4*B3 + 60A3*B4 + 90A4*B4,+,
      35A4*B3 + 35A3*B4 + 140A4*B4,+,
      70A4*B4}
  */

  const SCEVAddRecExpr *Product =
      dyn_cast<SCEVAddRecExpr>(SE.getMulExpr(A_rec, B_rec));
  ASSERT_TRUE(Product);
  ASSERT_EQ(Product->getNumOperands(), 9u);

  SmallVector<const SCEV *, 16> Sum;
  Sum.push_back(SE.getMulExpr(A[0], B[0]));
  EXPECT_EQ(Product->getOperand(0), SE.getAddExpr(Sum));
  Sum.clear();

  // SCEV produces different an equal but different expression for these.
  // Re-enable when PR11052 is fixed.
#if 0
  Sum.push_back(SE.getMulExpr(A[1], B[0]));
  Sum.push_back(SE.getMulExpr(A[0], B[1]));
  Sum.push_back(SE.getMulExpr(A[1], B[1]));
  EXPECT_EQ(Product->getOperand(1), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(A[2], B[0]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 2), A[1], B[1]));
  Sum.push_back(SE.getMulExpr(A[0], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 2), A[2], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 2), A[1], B[2]));
  Sum.push_back(SE.getMulExpr(A[2], B[2]));
  EXPECT_EQ(Product->getOperand(2), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(A[3], B[0]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[2], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[1], B[2]));
  Sum.push_back(SE.getMulExpr(A[0], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[3], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 6), A[2], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[1], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[3], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[2], B[3]));
  Sum.push_back(SE.getMulExpr(A[3], B[3]));
  EXPECT_EQ(Product->getOperand(3), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(A[4], B[0]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[3], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 6), A[2], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[1], B[3]));
  Sum.push_back(SE.getMulExpr(A[0], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[4], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 12), A[3], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 12), A[2], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[1], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 6), A[4], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 12), A[3], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 6), A[2], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[4], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[3], B[4]));
  Sum.push_back(SE.getMulExpr(A[4], B[4]));
  EXPECT_EQ(Product->getOperand(4), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 5), A[4], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 10), A[3], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 10), A[2], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 5), A[1], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 20), A[4], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 30), A[3], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 20), A[2], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 30), A[4], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 30), A[3], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 20), A[4], B[4]));
  EXPECT_EQ(Product->getOperand(5), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 15), A[4], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 20), A[3], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 15), A[2], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 60), A[4], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 60), A[3], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 90), A[4], B[4]));
  EXPECT_EQ(Product->getOperand(6), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 35), A[4], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 35), A[3], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 140), A[4], B[4]));
  EXPECT_EQ(Product->getOperand(7), SE.getAddExpr(Sum));
  Sum.clear();
#endif

  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 70), A[4], B[4]));
  EXPECT_EQ(Product->getOperand(8), SE.getAddExpr(Sum));
}

}  // end anonymous namespace
}  // end namespace llvm
