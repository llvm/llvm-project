//===- llvm/unittest/IR/InstructionsTest.cpp - Instructions unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include <memory>

namespace llvm {
namespace {

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("InstructionsTests", errs());
  return Mod;
}

TEST(InstructionsTest, ReturnInst) {
  LLVMContext C;

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

// Test fixture that provides a module and a single function within it. Useful
// for tests that need to refer to the function in some way.
class ModuleWithFunctionTest : public testing::Test {
protected:
  ModuleWithFunctionTest() : M(new Module("MyModule", Ctx)) {
    FArgTypes.push_back(Type::getInt8Ty(Ctx));
    FArgTypes.push_back(Type::getInt32Ty(Ctx));
    FArgTypes.push_back(Type::getInt64Ty(Ctx));
    FunctionType *FTy =
        FunctionType::get(Type::getVoidTy(Ctx), FArgTypes, false);
    F = Function::Create(FTy, Function::ExternalLinkage, "", M.get());
  }

  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  SmallVector<Type *, 3> FArgTypes;
  Function *F;
};

TEST_F(ModuleWithFunctionTest, CallInst) {
  Value *Args[] = {ConstantInt::get(Type::getInt8Ty(Ctx), 20),
                   ConstantInt::get(Type::getInt32Ty(Ctx), 9999),
                   ConstantInt::get(Type::getInt64Ty(Ctx), 42)};
  std::unique_ptr<CallInst> Call(CallInst::Create(F, Args));

  // Make sure iteration over a call's arguments works as expected.
  unsigned Idx = 0;
  for (Value *Arg : Call->arg_operands()) {
    EXPECT_EQ(FArgTypes[Idx], Arg->getType());
    EXPECT_EQ(Call->getArgOperand(Idx)->getType(), Arg->getType());
    Idx++;
  }
}

TEST_F(ModuleWithFunctionTest, InvokeInst) {
  BasicBlock *BB1 = BasicBlock::Create(Ctx, "", F);
  BasicBlock *BB2 = BasicBlock::Create(Ctx, "", F);

  Value *Args[] = {ConstantInt::get(Type::getInt8Ty(Ctx), 20),
                   ConstantInt::get(Type::getInt32Ty(Ctx), 9999),
                   ConstantInt::get(Type::getInt64Ty(Ctx), 42)};
  std::unique_ptr<InvokeInst> Invoke(InvokeInst::Create(F, BB1, BB2, Args));

  // Make sure iteration over invoke's arguments works as expected.
  unsigned Idx = 0;
  for (Value *Arg : Invoke->arg_operands()) {
    EXPECT_EQ(FArgTypes[Idx], Arg->getType());
    EXPECT_EQ(Invoke->getArgOperand(Idx)->getType(), Arg->getType());
    Idx++;
  }
}

TEST(InstructionsTest, BranchInst) {
  LLVMContext C;

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
  EXPECT_EQ(b0->op_end(), std::next(b0->op_begin()));

  EXPECT_EQ(b0->op_end(), std::next(b0->op_begin()));

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
  LLVMContext C;

  Type *Int8Ty = Type::getInt8Ty(C);
  Type *Int16Ty = Type::getInt16Ty(C);
  Type *Int32Ty = Type::getInt32Ty(C);
  Type *Int64Ty = Type::getInt64Ty(C);
  Type *V8x8Ty = VectorType::get(Int8Ty, 8);
  Type *V8x64Ty = VectorType::get(Int64Ty, 8);
  Type *X86MMXTy = Type::getX86_MMXTy(C);

  Type *HalfTy = Type::getHalfTy(C);
  Type *FloatTy = Type::getFloatTy(C);
  Type *DoubleTy = Type::getDoubleTy(C);

  Type *V2Int32Ty = VectorType::get(Int32Ty, 2);
  Type *V2Int64Ty = VectorType::get(Int64Ty, 2);
  Type *V4Int16Ty = VectorType::get(Int16Ty, 4);

  Type *Int32PtrTy = PointerType::get(Int32Ty, 0);
  Type *Int64PtrTy = PointerType::get(Int64Ty, 0);

  Type *Int32PtrAS1Ty = PointerType::get(Int32Ty, 1);
  Type *Int64PtrAS1Ty = PointerType::get(Int64Ty, 1);

  Type *V2Int32PtrAS1Ty = VectorType::get(Int32PtrAS1Ty, 2);
  Type *V2Int64PtrAS1Ty = VectorType::get(Int64PtrAS1Ty, 2);
  Type *V4Int32PtrAS1Ty = VectorType::get(Int32PtrAS1Ty, 4);
  Type *V4Int64PtrAS1Ty = VectorType::get(Int64PtrAS1Ty, 4);

  Type *V2Int64PtrTy = VectorType::get(Int64PtrTy, 2);
  Type *V2Int32PtrTy = VectorType::get(Int32PtrTy, 2);
  Type *V4Int32PtrTy = VectorType::get(Int32PtrTy, 4);

  const Constant* c8 = Constant::getNullValue(V8x8Ty);
  const Constant* c64 = Constant::getNullValue(V8x64Ty);

  const Constant *v2ptr32 = Constant::getNullValue(V2Int32PtrTy);

  EXPECT_TRUE(CastInst::isCastable(V8x8Ty, X86MMXTy));
  EXPECT_TRUE(CastInst::isCastable(X86MMXTy, V8x8Ty));
  EXPECT_FALSE(CastInst::isCastable(Int64Ty, X86MMXTy));
  EXPECT_TRUE(CastInst::isCastable(V8x64Ty, V8x8Ty));
  EXPECT_TRUE(CastInst::isCastable(V8x8Ty, V8x64Ty));
  EXPECT_EQ(CastInst::Trunc, CastInst::getCastOpcode(c64, true, V8x8Ty, true));
  EXPECT_EQ(CastInst::SExt, CastInst::getCastOpcode(c8, true, V8x64Ty, true));

  EXPECT_FALSE(CastInst::isBitCastable(V8x8Ty, X86MMXTy));
  EXPECT_FALSE(CastInst::isBitCastable(X86MMXTy, V8x8Ty));
  EXPECT_FALSE(CastInst::isBitCastable(Int64Ty, X86MMXTy));
  EXPECT_FALSE(CastInst::isBitCastable(V8x64Ty, V8x8Ty));
  EXPECT_FALSE(CastInst::isBitCastable(V8x8Ty, V8x64Ty));

  // Check address space casts are rejected since we don't know the sizes here
  EXPECT_FALSE(CastInst::isBitCastable(Int32PtrTy, Int32PtrAS1Ty));
  EXPECT_FALSE(CastInst::isBitCastable(Int32PtrAS1Ty, Int32PtrTy));
  EXPECT_FALSE(CastInst::isBitCastable(V2Int32PtrTy, V2Int32PtrAS1Ty));
  EXPECT_FALSE(CastInst::isBitCastable(V2Int32PtrAS1Ty, V2Int32PtrTy));
  EXPECT_TRUE(CastInst::isBitCastable(V2Int32PtrAS1Ty, V2Int64PtrAS1Ty));
  EXPECT_TRUE(CastInst::isCastable(V2Int32PtrAS1Ty, V2Int32PtrTy));
  EXPECT_EQ(CastInst::AddrSpaceCast, CastInst::getCastOpcode(v2ptr32, true,
                                                             V2Int32PtrAS1Ty,
                                                             true));

  // Test mismatched number of elements for pointers
  EXPECT_FALSE(CastInst::isBitCastable(V2Int32PtrAS1Ty, V4Int64PtrAS1Ty));
  EXPECT_FALSE(CastInst::isBitCastable(V4Int64PtrAS1Ty, V2Int32PtrAS1Ty));
  EXPECT_FALSE(CastInst::isBitCastable(V2Int32PtrAS1Ty, V4Int32PtrAS1Ty));
  EXPECT_FALSE(CastInst::isBitCastable(Int32PtrTy, V2Int32PtrTy));
  EXPECT_FALSE(CastInst::isBitCastable(V2Int32PtrTy, Int32PtrTy));

  EXPECT_TRUE(CastInst::isBitCastable(Int32PtrTy, Int64PtrTy));
  EXPECT_FALSE(CastInst::isBitCastable(DoubleTy, FloatTy));
  EXPECT_FALSE(CastInst::isBitCastable(FloatTy, DoubleTy));
  EXPECT_TRUE(CastInst::isBitCastable(FloatTy, FloatTy));
  EXPECT_TRUE(CastInst::isBitCastable(FloatTy, FloatTy));
  EXPECT_TRUE(CastInst::isBitCastable(FloatTy, Int32Ty));
  EXPECT_TRUE(CastInst::isBitCastable(Int16Ty, HalfTy));
  EXPECT_TRUE(CastInst::isBitCastable(Int32Ty, FloatTy));
  EXPECT_TRUE(CastInst::isBitCastable(V2Int32Ty, Int64Ty));

  EXPECT_TRUE(CastInst::isBitCastable(V2Int32Ty, V4Int16Ty));
  EXPECT_FALSE(CastInst::isBitCastable(Int32Ty, Int64Ty));
  EXPECT_FALSE(CastInst::isBitCastable(Int64Ty, Int32Ty));

  EXPECT_FALSE(CastInst::isBitCastable(V2Int32PtrTy, Int64Ty));
  EXPECT_FALSE(CastInst::isBitCastable(Int64Ty, V2Int32PtrTy));
  EXPECT_TRUE(CastInst::isBitCastable(V2Int64PtrTy, V2Int32PtrTy));
  EXPECT_TRUE(CastInst::isBitCastable(V2Int32PtrTy, V2Int64PtrTy));
  EXPECT_FALSE(CastInst::isBitCastable(V2Int32Ty, V2Int64Ty));
  EXPECT_FALSE(CastInst::isBitCastable(V2Int64Ty, V2Int32Ty));


  EXPECT_FALSE(CastInst::castIsValid(Instruction::BitCast,
                                     Constant::getNullValue(V4Int32PtrTy),
                                     V2Int32PtrTy));
  EXPECT_FALSE(CastInst::castIsValid(Instruction::BitCast,
                                     Constant::getNullValue(V2Int32PtrTy),
                                     V4Int32PtrTy));

  EXPECT_FALSE(CastInst::castIsValid(Instruction::AddrSpaceCast,
                                     Constant::getNullValue(V4Int32PtrAS1Ty),
                                     V2Int32PtrTy));
  EXPECT_FALSE(CastInst::castIsValid(Instruction::AddrSpaceCast,
                                     Constant::getNullValue(V2Int32PtrTy),
                                     V4Int32PtrAS1Ty));


  // Check that assertion is not hit when creating a cast with a vector of
  // pointers
  // First form
  BasicBlock *BB = BasicBlock::Create(C);
  Constant *NullV2I32Ptr = Constant::getNullValue(V2Int32PtrTy);
  auto Inst1 = CastInst::CreatePointerCast(NullV2I32Ptr, V2Int32Ty, "foo", BB);

  // Second form
  auto Inst2 = CastInst::CreatePointerCast(NullV2I32Ptr, V2Int32Ty);

  delete Inst2;
  Inst1->eraseFromParent();
  delete BB;
}

TEST(InstructionsTest, VectorGep) {
  LLVMContext C;

  // Type Definitions
  Type *I8Ty = IntegerType::get(C, 8);
  Type *I32Ty = IntegerType::get(C, 32);
  PointerType *Ptri8Ty = PointerType::get(I8Ty, 0);
  PointerType *Ptri32Ty = PointerType::get(I32Ty, 0);

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

  BasicBlock* BB0 = BasicBlock::Create(C);
  // Test InsertAtEnd ICmpInst constructor.
  ICmpInst *ICmp2 = new ICmpInst(*BB0, ICmpInst::ICMP_SGE, PtrVecA, PtrVecB);
  EXPECT_NE(ICmp0, ICmp2); // suppress warning.

  GetElementPtrInst *Gep0 = GetElementPtrInst::Create(I32Ty, PtrVecA, C2xi32a);
  GetElementPtrInst *Gep1 = GetElementPtrInst::Create(I32Ty, PtrVecA, C2xi32b);
  GetElementPtrInst *Gep2 = GetElementPtrInst::Create(I32Ty, PtrVecB, C2xi32a);
  GetElementPtrInst *Gep3 = GetElementPtrInst::Create(I32Ty, PtrVecB, C2xi32b);

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
                "2:32:32-f64:64:64-v64:64:64-v128:128:128-a:0:64-s:64:64-f80"
                ":128:128-n8:16:32:64-S128");
  // Make sure we don't crash
  GetPointerBaseWithConstantOffset(Gep0, Offset, TD);
  GetPointerBaseWithConstantOffset(Gep1, Offset, TD);
  GetPointerBaseWithConstantOffset(Gep2, Offset, TD);
  GetPointerBaseWithConstantOffset(Gep3, Offset, TD);

  // Gep of Geps
  GetElementPtrInst *GepII0 = GetElementPtrInst::Create(I32Ty, Gep0, C2xi32b);
  GetElementPtrInst *GepII1 = GetElementPtrInst::Create(I32Ty, Gep1, C2xi32a);
  GetElementPtrInst *GepII2 = GetElementPtrInst::Create(I32Ty, Gep2, C2xi32b);
  GetElementPtrInst *GepII3 = GetElementPtrInst::Create(I32Ty, Gep3, C2xi32a);

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

  ICmp2->eraseFromParent();
  delete BB0;

  delete ICmp0;
  delete ICmp1;
  delete PtrVecA;
  delete PtrVecB;
}

TEST(InstructionsTest, FPMathOperator) {
  LLVMContext Context;
  IRBuilder<> Builder(Context);
  MDBuilder MDHelper(Context);
  Instruction *I = Builder.CreatePHI(Builder.getDoubleTy(), 0);
  MDNode *MD1 = MDHelper.createFPMath(1.0);
  Value *V1 = Builder.CreateFAdd(I, I, "", MD1);
  EXPECT_TRUE(isa<FPMathOperator>(V1));
  FPMathOperator *O1 = cast<FPMathOperator>(V1);
  EXPECT_EQ(O1->getFPAccuracy(), 1.0);
  V1->deleteValue();
  I->deleteValue();
}


TEST(InstructionsTest, isEliminableCastPair) {
  LLVMContext C;

  Type* Int16Ty = Type::getInt16Ty(C);
  Type* Int32Ty = Type::getInt32Ty(C);
  Type* Int64Ty = Type::getInt64Ty(C);
  Type* Int64PtrTy = Type::getInt64PtrTy(C);

  // Source and destination pointers have same size -> bitcast.
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::PtrToInt,
                                           CastInst::IntToPtr,
                                           Int64PtrTy, Int64Ty, Int64PtrTy,
                                           Int32Ty, nullptr, Int32Ty),
            CastInst::BitCast);

  // Source and destination have unknown sizes, but the same address space and
  // the intermediate int is the maximum pointer size -> bitcast
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::PtrToInt,
                                           CastInst::IntToPtr,
                                           Int64PtrTy, Int64Ty, Int64PtrTy,
                                           nullptr, nullptr, nullptr),
            CastInst::BitCast);

  // Source and destination have unknown sizes, but the same address space and
  // the intermediate int is not the maximum pointer size -> nothing
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::PtrToInt,
                                           CastInst::IntToPtr,
                                           Int64PtrTy, Int32Ty, Int64PtrTy,
                                           nullptr, nullptr, nullptr),
            0U);

  // Middle pointer big enough -> bitcast.
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::IntToPtr,
                                           CastInst::PtrToInt,
                                           Int64Ty, Int64PtrTy, Int64Ty,
                                           nullptr, Int64Ty, nullptr),
            CastInst::BitCast);

  // Middle pointer too small -> fail.
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::IntToPtr,
                                           CastInst::PtrToInt,
                                           Int64Ty, Int64PtrTy, Int64Ty,
                                           nullptr, Int32Ty, nullptr),
            0U);

  // Test that we don't eliminate bitcasts between different address spaces,
  // or if we don't have available pointer size information.
  DataLayout DL("e-p:32:32:32-p1:16:16:16-p2:64:64:64-i1:8:8-i8:8:8-i16:16:16"
                "-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64"
                "-v128:128:128-a:0:64-s:64:64-f80:128:128-n8:16:32:64-S128");

  Type* Int64PtrTyAS1 = Type::getInt64PtrTy(C, 1);
  Type* Int64PtrTyAS2 = Type::getInt64PtrTy(C, 2);

  IntegerType *Int16SizePtr = DL.getIntPtrType(C, 1);
  IntegerType *Int64SizePtr = DL.getIntPtrType(C, 2);

  // Cannot simplify inttoptr, addrspacecast
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::IntToPtr,
                                           CastInst::AddrSpaceCast,
                                           Int16Ty, Int64PtrTyAS1, Int64PtrTyAS2,
                                           nullptr, Int16SizePtr, Int64SizePtr),
            0U);

  // Cannot simplify addrspacecast, ptrtoint
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::AddrSpaceCast,
                                           CastInst::PtrToInt,
                                           Int64PtrTyAS1, Int64PtrTyAS2, Int16Ty,
                                           Int64SizePtr, Int16SizePtr, nullptr),
            0U);

  // Pass since the bitcast address spaces are the same
  EXPECT_EQ(CastInst::isEliminableCastPair(CastInst::IntToPtr,
                                           CastInst::BitCast,
                                           Int16Ty, Int64PtrTyAS1, Int64PtrTyAS1,
                                           nullptr, nullptr, nullptr),
            CastInst::IntToPtr);

}

TEST(InstructionsTest, CloneCall) {
  LLVMContext C;
  Type *Int32Ty = Type::getInt32Ty(C);
  Type *ArgTys[] = {Int32Ty, Int32Ty, Int32Ty};
  FunctionType *FnTy = FunctionType::get(Int32Ty, ArgTys, /*isVarArg=*/false);
  Value *Callee = Constant::getNullValue(FnTy->getPointerTo());
  Value *Args[] = {
    ConstantInt::get(Int32Ty, 1),
    ConstantInt::get(Int32Ty, 2),
    ConstantInt::get(Int32Ty, 3)
  };
  std::unique_ptr<CallInst> Call(
      CallInst::Create(FnTy, Callee, Args, "result"));

  // Test cloning the tail call kind.
  CallInst::TailCallKind Kinds[] = {CallInst::TCK_None, CallInst::TCK_Tail,
                                    CallInst::TCK_MustTail};
  for (CallInst::TailCallKind TCK : Kinds) {
    Call->setTailCallKind(TCK);
    std::unique_ptr<CallInst> Clone(cast<CallInst>(Call->clone()));
    EXPECT_EQ(Call->getTailCallKind(), Clone->getTailCallKind());
  }
  Call->setTailCallKind(CallInst::TCK_None);

  // Test cloning an attribute.
  {
    AttrBuilder AB;
    AB.addAttribute(Attribute::ReadOnly);
    Call->setAttributes(
        AttributeList::get(C, AttributeList::FunctionIndex, AB));
    std::unique_ptr<CallInst> Clone(cast<CallInst>(Call->clone()));
    EXPECT_TRUE(Clone->onlyReadsMemory());
  }
}

TEST(InstructionsTest, AlterCallBundles) {
  LLVMContext C;
  Type *Int32Ty = Type::getInt32Ty(C);
  FunctionType *FnTy = FunctionType::get(Int32Ty, Int32Ty, /*isVarArg=*/false);
  Value *Callee = Constant::getNullValue(FnTy->getPointerTo());
  Value *Args[] = {ConstantInt::get(Int32Ty, 42)};
  OperandBundleDef OldBundle("before", UndefValue::get(Int32Ty));
  std::unique_ptr<CallInst> Call(
      CallInst::Create(FnTy, Callee, Args, OldBundle, "result"));
  Call->setTailCallKind(CallInst::TailCallKind::TCK_NoTail);
  AttrBuilder AB;
  AB.addAttribute(Attribute::Cold);
  Call->setAttributes(AttributeList::get(C, AttributeList::FunctionIndex, AB));
  Call->setDebugLoc(DebugLoc(MDNode::get(C, None)));

  OperandBundleDef NewBundle("after", ConstantInt::get(Int32Ty, 7));
  std::unique_ptr<CallInst> Clone(CallInst::Create(Call.get(), NewBundle));
  EXPECT_EQ(Call->getNumArgOperands(), Clone->getNumArgOperands());
  EXPECT_EQ(Call->getArgOperand(0), Clone->getArgOperand(0));
  EXPECT_EQ(Call->getCallingConv(), Clone->getCallingConv());
  EXPECT_EQ(Call->getTailCallKind(), Clone->getTailCallKind());
  EXPECT_TRUE(Clone->hasFnAttr(Attribute::AttrKind::Cold));
  EXPECT_EQ(Call->getDebugLoc(), Clone->getDebugLoc());
  EXPECT_EQ(Clone->getNumOperandBundles(), 1U);
  EXPECT_TRUE(Clone->getOperandBundle("after").hasValue());
}

TEST(InstructionsTest, AlterInvokeBundles) {
  LLVMContext C;
  Type *Int32Ty = Type::getInt32Ty(C);
  FunctionType *FnTy = FunctionType::get(Int32Ty, Int32Ty, /*isVarArg=*/false);
  Value *Callee = Constant::getNullValue(FnTy->getPointerTo());
  Value *Args[] = {ConstantInt::get(Int32Ty, 42)};
  std::unique_ptr<BasicBlock> NormalDest(BasicBlock::Create(C));
  std::unique_ptr<BasicBlock> UnwindDest(BasicBlock::Create(C));
  OperandBundleDef OldBundle("before", UndefValue::get(Int32Ty));
  std::unique_ptr<InvokeInst> Invoke(
      InvokeInst::Create(FnTy, Callee, NormalDest.get(), UnwindDest.get(), Args,
                         OldBundle, "result"));
  AttrBuilder AB;
  AB.addAttribute(Attribute::Cold);
  Invoke->setAttributes(
      AttributeList::get(C, AttributeList::FunctionIndex, AB));
  Invoke->setDebugLoc(DebugLoc(MDNode::get(C, None)));

  OperandBundleDef NewBundle("after", ConstantInt::get(Int32Ty, 7));
  std::unique_ptr<InvokeInst> Clone(
      InvokeInst::Create(Invoke.get(), NewBundle));
  EXPECT_EQ(Invoke->getNormalDest(), Clone->getNormalDest());
  EXPECT_EQ(Invoke->getUnwindDest(), Clone->getUnwindDest());
  EXPECT_EQ(Invoke->getNumArgOperands(), Clone->getNumArgOperands());
  EXPECT_EQ(Invoke->getArgOperand(0), Clone->getArgOperand(0));
  EXPECT_EQ(Invoke->getCallingConv(), Clone->getCallingConv());
  EXPECT_TRUE(Clone->hasFnAttr(Attribute::AttrKind::Cold));
  EXPECT_EQ(Invoke->getDebugLoc(), Clone->getDebugLoc());
  EXPECT_EQ(Clone->getNumOperandBundles(), 1U);
  EXPECT_TRUE(Clone->getOperandBundle("after").hasValue());
}

TEST_F(ModuleWithFunctionTest, DropPoisonGeneratingFlags) {
  auto *OnlyBB = BasicBlock::Create(Ctx, "bb", F);
  auto *Arg0 = &*F->arg_begin();

  IRBuilder<NoFolder> B(Ctx);
  B.SetInsertPoint(OnlyBB);

  {
    auto *UI =
        cast<Instruction>(B.CreateUDiv(Arg0, Arg0, "", /*isExact*/ true));
    ASSERT_TRUE(UI->isExact());
    UI->dropPoisonGeneratingFlags();
    ASSERT_FALSE(UI->isExact());
  }

  {
    auto *ShrI =
        cast<Instruction>(B.CreateLShr(Arg0, Arg0, "", /*isExact*/ true));
    ASSERT_TRUE(ShrI->isExact());
    ShrI->dropPoisonGeneratingFlags();
    ASSERT_FALSE(ShrI->isExact());
  }

  {
    auto *AI = cast<Instruction>(
        B.CreateAdd(Arg0, Arg0, "", /*HasNUW*/ true, /*HasNSW*/ false));
    ASSERT_TRUE(AI->hasNoUnsignedWrap());
    AI->dropPoisonGeneratingFlags();
    ASSERT_FALSE(AI->hasNoUnsignedWrap());
    ASSERT_FALSE(AI->hasNoSignedWrap());
  }

  {
    auto *SI = cast<Instruction>(
        B.CreateAdd(Arg0, Arg0, "", /*HasNUW*/ false, /*HasNSW*/ true));
    ASSERT_TRUE(SI->hasNoSignedWrap());
    SI->dropPoisonGeneratingFlags();
    ASSERT_FALSE(SI->hasNoUnsignedWrap());
    ASSERT_FALSE(SI->hasNoSignedWrap());
  }

  {
    auto *ShlI = cast<Instruction>(
        B.CreateShl(Arg0, Arg0, "", /*HasNUW*/ true, /*HasNSW*/ true));
    ASSERT_TRUE(ShlI->hasNoSignedWrap());
    ASSERT_TRUE(ShlI->hasNoUnsignedWrap());
    ShlI->dropPoisonGeneratingFlags();
    ASSERT_FALSE(ShlI->hasNoUnsignedWrap());
    ASSERT_FALSE(ShlI->hasNoSignedWrap());
  }

  {
    Value *GEPBase = Constant::getNullValue(B.getInt8PtrTy());
    auto *GI = cast<GetElementPtrInst>(
        B.CreateInBoundsGEP(B.getInt8Ty(), GEPBase, Arg0));
    ASSERT_TRUE(GI->isInBounds());
    GI->dropPoisonGeneratingFlags();
    ASSERT_FALSE(GI->isInBounds());
  }
}

TEST(InstructionsTest, GEPIndices) {
  LLVMContext Context;
  IRBuilder<NoFolder> Builder(Context);
  Type *ElementTy = Builder.getInt8Ty();
  Type *ArrTy = ArrayType::get(ArrayType::get(ElementTy, 64), 64);
  Value *Indices[] = {
    Builder.getInt32(0),
    Builder.getInt32(13),
    Builder.getInt32(42) };

  Value *V = Builder.CreateGEP(ArrTy, UndefValue::get(PointerType::getUnqual(ArrTy)),
                               Indices);
  ASSERT_TRUE(isa<GetElementPtrInst>(V));

  auto *GEPI = cast<GetElementPtrInst>(V);
  ASSERT_NE(GEPI->idx_begin(), GEPI->idx_end());
  ASSERT_EQ(GEPI->idx_end(), std::next(GEPI->idx_begin(), 3));
  EXPECT_EQ(Indices[0], GEPI->idx_begin()[0]);
  EXPECT_EQ(Indices[1], GEPI->idx_begin()[1]);
  EXPECT_EQ(Indices[2], GEPI->idx_begin()[2]);
  EXPECT_EQ(GEPI->idx_begin(), GEPI->indices().begin());
  EXPECT_EQ(GEPI->idx_end(), GEPI->indices().end());

  const auto *CGEPI = GEPI;
  ASSERT_NE(CGEPI->idx_begin(), CGEPI->idx_end());
  ASSERT_EQ(CGEPI->idx_end(), std::next(CGEPI->idx_begin(), 3));
  EXPECT_EQ(Indices[0], CGEPI->idx_begin()[0]);
  EXPECT_EQ(Indices[1], CGEPI->idx_begin()[1]);
  EXPECT_EQ(Indices[2], CGEPI->idx_begin()[2]);
  EXPECT_EQ(CGEPI->idx_begin(), CGEPI->indices().begin());
  EXPECT_EQ(CGEPI->idx_end(), CGEPI->indices().end());

  delete GEPI;
}

TEST(InstructionsTest, SwitchInst) {
  LLVMContext C;

  std::unique_ptr<BasicBlock> BB1, BB2, BB3;
  BB1.reset(BasicBlock::Create(C));
  BB2.reset(BasicBlock::Create(C));
  BB3.reset(BasicBlock::Create(C));

  // We create block 0 after the others so that it gets destroyed first and
  // clears the uses of the other basic blocks.
  std::unique_ptr<BasicBlock> BB0(BasicBlock::Create(C));

  auto *Int32Ty = Type::getInt32Ty(C);

  SwitchInst *SI =
      SwitchInst::Create(UndefValue::get(Int32Ty), BB0.get(), 3, BB0.get());
  SI->addCase(ConstantInt::get(Int32Ty, 1), BB1.get());
  SI->addCase(ConstantInt::get(Int32Ty, 2), BB2.get());
  SI->addCase(ConstantInt::get(Int32Ty, 3), BB3.get());

  auto CI = SI->case_begin();
  ASSERT_NE(CI, SI->case_end());
  EXPECT_EQ(1, CI->getCaseValue()->getSExtValue());
  EXPECT_EQ(BB1.get(), CI->getCaseSuccessor());
  EXPECT_EQ(2, (CI + 1)->getCaseValue()->getSExtValue());
  EXPECT_EQ(BB2.get(), (CI + 1)->getCaseSuccessor());
  EXPECT_EQ(3, (CI + 2)->getCaseValue()->getSExtValue());
  EXPECT_EQ(BB3.get(), (CI + 2)->getCaseSuccessor());
  EXPECT_EQ(CI + 1, std::next(CI));
  EXPECT_EQ(CI + 2, std::next(CI, 2));
  EXPECT_EQ(CI + 3, std::next(CI, 3));
  EXPECT_EQ(SI->case_end(), CI + 3);
  EXPECT_EQ(0, CI - CI);
  EXPECT_EQ(1, (CI + 1) - CI);
  EXPECT_EQ(2, (CI + 2) - CI);
  EXPECT_EQ(3, SI->case_end() - CI);
  EXPECT_EQ(3, std::distance(CI, SI->case_end()));

  auto CCI = const_cast<const SwitchInst *>(SI)->case_begin();
  SwitchInst::ConstCaseIt CCE = SI->case_end();
  ASSERT_NE(CCI, SI->case_end());
  EXPECT_EQ(1, CCI->getCaseValue()->getSExtValue());
  EXPECT_EQ(BB1.get(), CCI->getCaseSuccessor());
  EXPECT_EQ(2, (CCI + 1)->getCaseValue()->getSExtValue());
  EXPECT_EQ(BB2.get(), (CCI + 1)->getCaseSuccessor());
  EXPECT_EQ(3, (CCI + 2)->getCaseValue()->getSExtValue());
  EXPECT_EQ(BB3.get(), (CCI + 2)->getCaseSuccessor());
  EXPECT_EQ(CCI + 1, std::next(CCI));
  EXPECT_EQ(CCI + 2, std::next(CCI, 2));
  EXPECT_EQ(CCI + 3, std::next(CCI, 3));
  EXPECT_EQ(CCE, CCI + 3);
  EXPECT_EQ(0, CCI - CCI);
  EXPECT_EQ(1, (CCI + 1) - CCI);
  EXPECT_EQ(2, (CCI + 2) - CCI);
  EXPECT_EQ(3, CCE - CCI);
  EXPECT_EQ(3, std::distance(CCI, CCE));

  // Make sure that the const iterator is compatible with a const auto ref.
  const auto &Handle = *CCI;
  EXPECT_EQ(1, Handle.getCaseValue()->getSExtValue());
  EXPECT_EQ(BB1.get(), Handle.getCaseSuccessor());
}

TEST(InstructionsTest, CommuteShuffleMask) {
  SmallVector<int, 16> Indices({-1, 0, 7});
  ShuffleVectorInst::commuteShuffleMask(Indices, 4);
  EXPECT_THAT(Indices, testing::ContainerEq(ArrayRef<int>({-1, 4, 3})));
}

TEST(InstructionsTest, ShuffleMaskQueries) {
  // Create the elements for various constant vectors.
  LLVMContext Ctx;
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Constant *CU = UndefValue::get(Int32Ty);
  Constant *C0 = ConstantInt::get(Int32Ty, 0);
  Constant *C1 = ConstantInt::get(Int32Ty, 1);
  Constant *C2 = ConstantInt::get(Int32Ty, 2);
  Constant *C3 = ConstantInt::get(Int32Ty, 3);
  Constant *C4 = ConstantInt::get(Int32Ty, 4);
  Constant *C5 = ConstantInt::get(Int32Ty, 5);
  Constant *C6 = ConstantInt::get(Int32Ty, 6);
  Constant *C7 = ConstantInt::get(Int32Ty, 7);

  Constant *Identity = ConstantVector::get({C0, CU, C2, C3, C4});
  EXPECT_TRUE(ShuffleVectorInst::isIdentityMask(Identity));
  EXPECT_FALSE(ShuffleVectorInst::isSelectMask(Identity)); // identity is distinguished from select
  EXPECT_FALSE(ShuffleVectorInst::isReverseMask(Identity));
  EXPECT_TRUE(ShuffleVectorInst::isSingleSourceMask(Identity)); // identity is always single source
  EXPECT_FALSE(ShuffleVectorInst::isZeroEltSplatMask(Identity));
  EXPECT_FALSE(ShuffleVectorInst::isTransposeMask(Identity));

  Constant *Select = ConstantVector::get({CU, C1, C5});
  EXPECT_FALSE(ShuffleVectorInst::isIdentityMask(Select));
  EXPECT_TRUE(ShuffleVectorInst::isSelectMask(Select));
  EXPECT_FALSE(ShuffleVectorInst::isReverseMask(Select));
  EXPECT_FALSE(ShuffleVectorInst::isSingleSourceMask(Select));
  EXPECT_FALSE(ShuffleVectorInst::isZeroEltSplatMask(Select));
  EXPECT_FALSE(ShuffleVectorInst::isTransposeMask(Select));
  
  Constant *Reverse = ConstantVector::get({C3, C2, C1, CU});
  EXPECT_FALSE(ShuffleVectorInst::isIdentityMask(Reverse));
  EXPECT_FALSE(ShuffleVectorInst::isSelectMask(Reverse));
  EXPECT_TRUE(ShuffleVectorInst::isReverseMask(Reverse));
  EXPECT_TRUE(ShuffleVectorInst::isSingleSourceMask(Reverse)); // reverse is always single source
  EXPECT_FALSE(ShuffleVectorInst::isZeroEltSplatMask(Reverse));
  EXPECT_FALSE(ShuffleVectorInst::isTransposeMask(Reverse));

  Constant *SingleSource = ConstantVector::get({C2, C2, C0, CU});
  EXPECT_FALSE(ShuffleVectorInst::isIdentityMask(SingleSource));
  EXPECT_FALSE(ShuffleVectorInst::isSelectMask(SingleSource));
  EXPECT_FALSE(ShuffleVectorInst::isReverseMask(SingleSource));
  EXPECT_TRUE(ShuffleVectorInst::isSingleSourceMask(SingleSource));
  EXPECT_FALSE(ShuffleVectorInst::isZeroEltSplatMask(SingleSource));
  EXPECT_FALSE(ShuffleVectorInst::isTransposeMask(SingleSource));

  Constant *ZeroEltSplat = ConstantVector::get({C0, C0, CU, C0});
  EXPECT_FALSE(ShuffleVectorInst::isIdentityMask(ZeroEltSplat));
  EXPECT_FALSE(ShuffleVectorInst::isSelectMask(ZeroEltSplat));
  EXPECT_FALSE(ShuffleVectorInst::isReverseMask(ZeroEltSplat));
  EXPECT_TRUE(ShuffleVectorInst::isSingleSourceMask(ZeroEltSplat)); // 0-splat is always single source
  EXPECT_TRUE(ShuffleVectorInst::isZeroEltSplatMask(ZeroEltSplat));
  EXPECT_FALSE(ShuffleVectorInst::isTransposeMask(ZeroEltSplat));

  Constant *Transpose = ConstantVector::get({C0, C4, C2, C6});
  EXPECT_FALSE(ShuffleVectorInst::isIdentityMask(Transpose));
  EXPECT_FALSE(ShuffleVectorInst::isSelectMask(Transpose));
  EXPECT_FALSE(ShuffleVectorInst::isReverseMask(Transpose));
  EXPECT_FALSE(ShuffleVectorInst::isSingleSourceMask(Transpose));
  EXPECT_FALSE(ShuffleVectorInst::isZeroEltSplatMask(Transpose));
  EXPECT_TRUE(ShuffleVectorInst::isTransposeMask(Transpose));

  // More tests to make sure the logic is/stays correct...
  EXPECT_TRUE(ShuffleVectorInst::isIdentityMask(ConstantVector::get({CU, C1, CU, C3})));
  EXPECT_TRUE(ShuffleVectorInst::isIdentityMask(ConstantVector::get({C4, CU, C6, CU})));

  EXPECT_TRUE(ShuffleVectorInst::isSelectMask(ConstantVector::get({C4, C1, C6, CU})));
  EXPECT_TRUE(ShuffleVectorInst::isSelectMask(ConstantVector::get({CU, C1, C6, C3})));

  EXPECT_TRUE(ShuffleVectorInst::isReverseMask(ConstantVector::get({C7, C6, CU, C4})));
  EXPECT_TRUE(ShuffleVectorInst::isReverseMask(ConstantVector::get({C3, CU, C1, CU})));

  EXPECT_TRUE(ShuffleVectorInst::isSingleSourceMask(ConstantVector::get({C7, C5, CU, C7})));
  EXPECT_TRUE(ShuffleVectorInst::isSingleSourceMask(ConstantVector::get({C3, C0, CU, C3})));

  EXPECT_TRUE(ShuffleVectorInst::isZeroEltSplatMask(ConstantVector::get({C4, CU, CU, C4})));
  EXPECT_TRUE(ShuffleVectorInst::isZeroEltSplatMask(ConstantVector::get({CU, C0, CU, C0})));

  EXPECT_TRUE(ShuffleVectorInst::isTransposeMask(ConstantVector::get({C1, C5, C3, C7})));
  EXPECT_TRUE(ShuffleVectorInst::isTransposeMask(ConstantVector::get({C1, C3})));

  // Nothing special about the values here - just re-using inputs to reduce code. 
  Constant *V0 = ConstantVector::get({C0, C1, C2, C3});
  Constant *V1 = ConstantVector::get({C3, C2, C1, C0});

  // Identity with undef elts.
  ShuffleVectorInst *Id1 = new ShuffleVectorInst(V0, V1,
                                                 ConstantVector::get({C0, C1, CU, CU}));
  EXPECT_TRUE(Id1->isIdentity());
  EXPECT_FALSE(Id1->isIdentityWithPadding());
  EXPECT_FALSE(Id1->isIdentityWithExtract());
  EXPECT_FALSE(Id1->isConcat());
  delete Id1;

  // Result has less elements than operands.
  ShuffleVectorInst *Id2 = new ShuffleVectorInst(V0, V1,
                                                 ConstantVector::get({C0, C1, C2}));
  EXPECT_FALSE(Id2->isIdentity());
  EXPECT_FALSE(Id2->isIdentityWithPadding());
  EXPECT_TRUE(Id2->isIdentityWithExtract());
  EXPECT_FALSE(Id2->isConcat());
  delete Id2;

  // Result has less elements than operands; choose from Op1.
  ShuffleVectorInst *Id3 = new ShuffleVectorInst(V0, V1,
                                                 ConstantVector::get({C4, CU, C6}));
  EXPECT_FALSE(Id3->isIdentity());
  EXPECT_FALSE(Id3->isIdentityWithPadding());
  EXPECT_TRUE(Id3->isIdentityWithExtract());
  EXPECT_FALSE(Id3->isConcat());
  delete Id3;

  // Result has less elements than operands; choose from Op0 and Op1 is not identity.
  ShuffleVectorInst *Id4 = new ShuffleVectorInst(V0, V1,
                                                 ConstantVector::get({C4, C1, C6}));
  EXPECT_FALSE(Id4->isIdentity());
  EXPECT_FALSE(Id4->isIdentityWithPadding());
  EXPECT_FALSE(Id4->isIdentityWithExtract());
  EXPECT_FALSE(Id4->isConcat());
  delete Id4;

  // Result has more elements than operands, and extra elements are undef.
  ShuffleVectorInst *Id5 = new ShuffleVectorInst(V0, V1,
                                                 ConstantVector::get({CU, C1, C2, C3, CU, CU}));
  EXPECT_FALSE(Id5->isIdentity());
  EXPECT_TRUE(Id5->isIdentityWithPadding());
  EXPECT_FALSE(Id5->isIdentityWithExtract());
  EXPECT_FALSE(Id5->isConcat());
  delete Id5;

  // Result has more elements than operands, and extra elements are undef; choose from Op1.
  ShuffleVectorInst *Id6 = new ShuffleVectorInst(V0, V1,
                                                 ConstantVector::get({C4, C5, C6, CU, CU, CU}));
  EXPECT_FALSE(Id6->isIdentity());
  EXPECT_TRUE(Id6->isIdentityWithPadding());
  EXPECT_FALSE(Id6->isIdentityWithExtract());
  EXPECT_FALSE(Id6->isConcat());
  delete Id6;
  
  // Result has more elements than operands, but extra elements are not undef.
  ShuffleVectorInst *Id7 = new ShuffleVectorInst(V0, V1,
                                                 ConstantVector::get({C0, C1, C2, C3, CU, C1}));
  EXPECT_FALSE(Id7->isIdentity());
  EXPECT_FALSE(Id7->isIdentityWithPadding());
  EXPECT_FALSE(Id7->isIdentityWithExtract());
  EXPECT_FALSE(Id7->isConcat());
  delete Id7;
  
  // Result has more elements than operands; choose from Op0 and Op1 is not identity.
  ShuffleVectorInst *Id8 = new ShuffleVectorInst(V0, V1,
                                                 ConstantVector::get({C4, CU, C2, C3, CU, CU}));
  EXPECT_FALSE(Id8->isIdentity());
  EXPECT_FALSE(Id8->isIdentityWithPadding());
  EXPECT_FALSE(Id8->isIdentityWithExtract());
  EXPECT_FALSE(Id8->isConcat());
  delete Id8;

  // Result has twice as many elements as operands; choose consecutively from Op0 and Op1 is concat.
  ShuffleVectorInst *Id9 = new ShuffleVectorInst(V0, V1,
                                                 ConstantVector::get({C0, CU, C2, C3, CU, CU, C6, C7}));
  EXPECT_FALSE(Id9->isIdentity());
  EXPECT_FALSE(Id9->isIdentityWithPadding());
  EXPECT_FALSE(Id9->isIdentityWithExtract());
  EXPECT_TRUE(Id9->isConcat());
  delete Id9;

  // Result has less than twice as many elements as operands, so not a concat.
  ShuffleVectorInst *Id10 = new ShuffleVectorInst(V0, V1,
                                                  ConstantVector::get({C0, CU, C2, C3, CU, CU, C6}));
  EXPECT_FALSE(Id10->isIdentity());
  EXPECT_FALSE(Id10->isIdentityWithPadding());
  EXPECT_FALSE(Id10->isIdentityWithExtract());
  EXPECT_FALSE(Id10->isConcat());
  delete Id10;

  // Result has more than twice as many elements as operands, so not a concat.
  ShuffleVectorInst *Id11 = new ShuffleVectorInst(V0, V1,
                                                  ConstantVector::get({C0, CU, C2, C3, CU, CU, C6, C7, CU}));
  EXPECT_FALSE(Id11->isIdentity());
  EXPECT_FALSE(Id11->isIdentityWithPadding());
  EXPECT_FALSE(Id11->isIdentityWithExtract());
  EXPECT_FALSE(Id11->isConcat());
  delete Id11;

  // If an input is undef, it's not a concat.
  // TODO: IdentityWithPadding should be true here even though the high mask values are not undef.
  ShuffleVectorInst *Id12 = new ShuffleVectorInst(V0, ConstantVector::get({CU, CU, CU, CU}),
                                                  ConstantVector::get({C0, CU, C2, C3, CU, CU, C6, C7}));
  EXPECT_FALSE(Id12->isIdentity());
  EXPECT_FALSE(Id12->isIdentityWithPadding());
  EXPECT_FALSE(Id12->isIdentityWithExtract());
  EXPECT_FALSE(Id12->isConcat());
  delete Id12;
}

TEST(InstructionsTest, SkipDebug) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"(
      declare void @llvm.dbg.value(metadata, metadata, metadata)

      define void @f() {
      entry:
        call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !13
        ret void
      }

      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3, !4}
      !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
      !1 = !DIFile(filename: "t2.c", directory: "foo")
      !2 = !{}
      !3 = !{i32 2, !"Dwarf Version", i32 4}
      !4 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
      !9 = !DISubroutineType(types: !10)
      !10 = !{null}
      !11 = !DILocalVariable(name: "x", scope: !8, file: !1, line: 2, type: !12)
      !12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !13 = !DILocation(line: 2, column: 7, scope: !8)
  )");
  ASSERT_TRUE(M);
  Function *F = cast<Function>(M->getNamedValue("f"));
  BasicBlock &BB = F->front();

  // The first non-debug instruction is the terminator.
  auto *Term = BB.getTerminator();
  EXPECT_EQ(Term, BB.begin()->getNextNonDebugInstruction());
  EXPECT_EQ(Term->getIterator(), skipDebugIntrinsics(BB.begin()));

  // After the terminator, there are no non-debug instructions.
  EXPECT_EQ(nullptr, Term->getNextNonDebugInstruction());
}

} // end anonymous namespace
} // end namespace llvm
