//===- llvm/unittest/VMCore/InstructionsTest.cpp - Instructions unit tests ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DataLayout.h"
#include "llvm/DerivedTypes.h"
#include "llvm/IRBuilder.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/MDBuilder.h"
#include "llvm/Operator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(InstructionsTest, ReturnInst) {
  LLVMContext &C(getGlobalContext());

  // test for PR6589
  const ReturnInst* r0 = ReturnInst::Create(C);
  EXPECT_EQ(r0->getNumOperands(), 0U);
  EXPECT_EQ(r0->op_begin(), r0->op_end());

  IntegerType* Int1 = IntegerType::get(C, 1);
  Constant* One = ConstantInt::get(Int1, 1, true);
  const ReturnInst* r1 = ReturnInst::Create(C, One);
  EXPECT_EQ(1U, r1->getNumOperands());
  User::const_op_iterator b(r1->op_begin());
  EXPECT_NE(r1->op_end(), b);
  EXPECT_EQ(One, *b);
  EXPECT_EQ(One, r1->getOperand(0));
  ++b;
  EXPECT_EQ(r1->op_end(), b);

  // clean up
  delete r0;
  delete r1;
}

TEST(InstructionsTest, BranchInst) {
  LLVMContext &C(getGlobalContext());

  // Make a BasicBlocks
  BasicBlock* bb0 = BasicBlock::Create(C);
  BasicBlock* bb1 = BasicBlock::Create(C);

  // Mandatory BranchInst
  const BranchInst* b0 = BranchInst::Create(bb0);

  EXPECT_TRUE(b0->isUnconditional());
  EXPECT_FALSE(b0->isConditional());
  EXPECT_EQ(1U, b0->getNumSuccessors());

  // check num operands
  EXPECT_EQ(1U, b0->getNumOperands());

  EXPECT_NE(b0->op_begin(), b0->op_end());
  EXPECT_EQ(b0->op_end(), llvm::next(b0->op_begin()));

  EXPECT_EQ(b0->op_end(), llvm::next(b0->op_begin()));

  IntegerType* Int1 = IntegerType::get(C, 1);
  Constant* One = ConstantInt::get(Int1, 1, true);

  // Conditional BranchInst
  BranchInst* b1 = BranchInst::Create(bb0, bb1, One);

  EXPECT_FALSE(b1->isUnconditional());
  EXPECT_TRUE(b1->isConditional());
  EXPECT_EQ(2U, b1->getNumSuccessors());

  // check num operands
  EXPECT_EQ(3U, b1->getNumOperands());

  User::const_op_iterator b(b1->op_begin());

  // check COND
  EXPECT_NE(b, b1->op_end());
  EXPECT_EQ(One, *b);
  EXPECT_EQ(One, b1->getOperand(0));
  EXPECT_EQ(One, b1->getCondition());
  ++b;

  // check ELSE
  EXPECT_EQ(bb1, *b);
  EXPECT_EQ(bb1, b1->getOperand(1));
  EXPECT_EQ(bb1, b1->getSuccessor(1));
  ++b;

  // check THEN
  EXPECT_EQ(bb0, *b);
  EXPECT_EQ(bb0, b1->getOperand(2));
  EXPECT_EQ(bb0, b1->getSuccessor(0));
  ++b;

  EXPECT_EQ(b1->op_end(), b);

  // clean up
  delete b0;
  delete b1;

  delete bb0;
  delete bb1;
}

TEST(InstructionsTest, CastInst) {
  LLVMContext &C(getGlobalContext());

  Type* Int8Ty = Type::getInt8Ty(C);
  Type* Int64Ty = Type::getInt64Ty(C);
  Type* V8x8Ty = VectorType::get(Int8Ty, 8);
  Type* V8x64Ty = VectorType::get(Int64Ty, 8);
  Type* X86MMXTy = Type::getX86_MMXTy(C);

  const Constant* c8 = Constant::getNullValue(V8x8Ty);
  const Constant* c64 = Constant::getNullValue(V8x64Ty);

  EXPECT_TRUE(CastInst::isCastable(V8x8Ty, X86MMXTy));
  EXPECT_TRUE(CastInst::isCastable(X86MMXTy, V8x8Ty));
  EXPECT_FALSE(CastInst::isCastable(Int64Ty, X86MMXTy));
  EXPECT_TRUE(CastInst::isCastable(V8x64Ty, V8x8Ty));
  EXPECT_TRUE(CastInst::isCastable(V8x8Ty, V8x64Ty));
  EXPECT_EQ(CastInst::Trunc, CastInst::getCastOpcode(c64, true, V8x8Ty, true));
  EXPECT_EQ(CastInst::SExt, CastInst::getCastOpcode(c8, true, V8x64Ty, true));
}



TEST(InstructionsTest, VectorGep) {
  LLVMContext &C(getGlobalContext());

  // Type Definitions
  PointerType *Ptri8Ty = PointerType::get(IntegerType::get(C, 8), 0);
  PointerType *Ptri32Ty = PointerType::get(IntegerType::get(C, 8), 0);

  VectorType *V2xi8PTy = VectorType::get(Ptri8Ty, 2);
  VectorType *V2xi32PTy = VectorType::get(Ptri32Ty, 2);

  // Test different aspects of the vector-of-pointers type
  // and GEPs which use this type.
  ConstantInt *Ci32a = ConstantInt::get(C, APInt(32, 1492));
  ConstantInt *Ci32b = ConstantInt::get(C, APInt(32, 1948));
  std::vector<Constant*> ConstVa(2, Ci32a);
  std::vector<Constant*> ConstVb(2, Ci32b);
  Constant *C2xi32a = ConstantVector::get(ConstVa);
  Constant *C2xi32b = ConstantVector::get(ConstVb);

  CastInst *PtrVecA = new IntToPtrInst(C2xi32a, V2xi32PTy);
  CastInst *PtrVecB = new IntToPtrInst(C2xi32b, V2xi32PTy);

  ICmpInst *ICmp0 = new ICmpInst(ICmpInst::ICMP_SGT, PtrVecA, PtrVecB);
  ICmpInst *ICmp1 = new ICmpInst(ICmpInst::ICMP_ULT, PtrVecA, PtrVecB);
  EXPECT_NE(ICmp0, ICmp1); // suppress warning.

  GetElementPtrInst *Gep0 = GetElementPtrInst::Create(PtrVecA, C2xi32a);
  GetElementPtrInst *Gep1 = GetElementPtrInst::Create(PtrVecA, C2xi32b);
  GetElementPtrInst *Gep2 = GetElementPtrInst::Create(PtrVecB, C2xi32a);
  GetElementPtrInst *Gep3 = GetElementPtrInst::Create(PtrVecB, C2xi32b);

  CastInst *BTC0 = new BitCastInst(Gep0, V2xi8PTy);
  CastInst *BTC1 = new BitCastInst(Gep1, V2xi8PTy);
  CastInst *BTC2 = new BitCastInst(Gep2, V2xi8PTy);
  CastInst *BTC3 = new BitCastInst(Gep3, V2xi8PTy);

  Value *S0 = BTC0->stripPointerCasts();
  Value *S1 = BTC1->stripPointerCasts();
  Value *S2 = BTC2->stripPointerCasts();
  Value *S3 = BTC3->stripPointerCasts();

  EXPECT_NE(S0, Gep0);
  EXPECT_NE(S1, Gep1);
  EXPECT_NE(S2, Gep2);
  EXPECT_NE(S3, Gep3);

  int64_t Offset;
  DataLayout TD("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f3"
                "2:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80"
                ":128:128-n8:16:32:64-S128");
  // Make sure we don't crash
  GetPointerBaseWithConstantOffset(Gep0, Offset, TD);
  GetPointerBaseWithConstantOffset(Gep1, Offset, TD);
  GetPointerBaseWithConstantOffset(Gep2, Offset, TD);
  GetPointerBaseWithConstantOffset(Gep3, Offset, TD);

  // Gep of Geps
  GetElementPtrInst *GepII0 = GetElementPtrInst::Create(Gep0, C2xi32b);
  GetElementPtrInst *GepII1 = GetElementPtrInst::Create(Gep1, C2xi32a);
  GetElementPtrInst *GepII2 = GetElementPtrInst::Create(Gep2, C2xi32b);
  GetElementPtrInst *GepII3 = GetElementPtrInst::Create(Gep3, C2xi32a);

  EXPECT_EQ(GepII0->getNumIndices(), 1u);
  EXPECT_EQ(GepII1->getNumIndices(), 1u);
  EXPECT_EQ(GepII2->getNumIndices(), 1u);
  EXPECT_EQ(GepII3->getNumIndices(), 1u);

  EXPECT_FALSE(GepII0->hasAllZeroIndices());
  EXPECT_FALSE(GepII1->hasAllZeroIndices());
  EXPECT_FALSE(GepII2->hasAllZeroIndices());
  EXPECT_FALSE(GepII3->hasAllZeroIndices());

  delete GepII0;
  delete GepII1;
  delete GepII2;
  delete GepII3;

  delete BTC0;
  delete BTC1;
  delete BTC2;
  delete BTC3;

  delete Gep0;
  delete Gep1;
  delete Gep2;
  delete Gep3;

  delete ICmp0;
  delete ICmp1;
  delete PtrVecA;
  delete PtrVecB;
}

TEST(InstructionsTest, FPMathOperator) {
  LLVMContext &Context = getGlobalContext();
  IRBuilder<> Builder(Context);
  MDBuilder MDHelper(Context);
  Instruction *I = Builder.CreatePHI(Builder.getDoubleTy(), 0);
  MDNode *MD1 = MDHelper.createFPMath(1.0);
  Value *V1 = Builder.CreateFAdd(I, I, "", MD1);
  EXPECT_TRUE(isa<FPMathOperator>(V1));
  FPMathOperator *O1 = cast<FPMathOperator>(V1);
  EXPECT_EQ(O1->getFPAccuracy(), 1.0);
  delete V1;
  delete I;
}


TEST(InstructionsTest, isEliminableCastPair) {
  LLVMContext &C(getGlobalContext());

  Type* Int32Ty = Type::getInt32Ty(C);
  Type* Int64Ty = Type::getInt64Ty(C);
  Type* Int64PtrTy = Type::getInt64PtrTy(C);

  // Source and destination pointers have same size -> bitcast.
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::PtrToInt,
                                           CastInst::IntToPtr,
                                           Int64PtrTy, Int64Ty, Int64PtrTy,
                                           Int32Ty, 0, Int32Ty),
            CastInst::BitCast);

  // Source and destination pointers have different sizes -> fail.
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::PtrToInt,
                                           CastInst::IntToPtr,
                                           Int64PtrTy, Int64Ty, Int64PtrTy,
                                           Int32Ty, 0, Int64Ty),
            0U);

  // Middle pointer big enough -> bitcast.
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::IntToPtr,
                                           CastInst::PtrToInt,
                                           Int64Ty, Int64PtrTy, Int64Ty,
                                           0, Int64Ty, 0),
            CastInst::BitCast);

  // Middle pointer too small -> fail.
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::IntToPtr,
                                           CastInst::PtrToInt,
                                           Int64Ty, Int64PtrTy, Int64Ty,
                                           0, Int32Ty, 0),
            0U);
}

}  // end anonymous namespace
}  // end namespace llvm
