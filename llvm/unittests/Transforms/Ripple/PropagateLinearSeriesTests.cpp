//===- PropagateLinearSeriesTests.cpp - -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RippleTestBase.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsRipple.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Ripple/Ripple.h"
#include "gtest/gtest.h"
#include <string>

namespace llvm {
namespace {

using PropagateLinearSeriesTest = RippleFunctionTest;

TEST_F(PropagateLinearSeriesTest, RippleBlockIndex) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [8][4] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(8);
  Value *Dim1 = IRB.getInt64(4);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  // Create ripple_block_index for dimension 0 -> [8]
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  // Create ripple_block_index for dimension 1 -> [1][4]
  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 = IRB.CreateCall(IndexFn, {SetShape, Idx1Val});

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Get LinearSeries for Index0
  const LinearSeries *LS0 = Ripple->getLinseriesFor(Index0);
  ASSERT_NE(LS0, nullptr);

  // Verify Index0 LinearSeries: base = 0, slope[0] = 1, slope[1] = 0
  EXPECT_TRUE(LS0->getBaseShape().isScalar());
  EXPECT_EQ(LS0->getSlopeShape().rank(), 2u);
  EXPECT_EQ(LS0->getSlopeShape()[0], 8u);
  EXPECT_EQ(LS0->getSlopeShape()[1], 1u);

  // Base should be constant 0
  if (auto *BaseConst = dyn_cast<ConstantInt>(LS0->getBase())) {
    EXPECT_EQ(BaseConst->getZExtValue(), 0u);
  } else {
    FAIL() << "Base is not a ConstantInt";
  }

  // Slope[0] should be constant 1
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS0->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 1u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Slope[1] should be constant 0
  if (auto *Slope1Const = dyn_cast<ConstantInt>(LS0->getSlope(1))) {
    EXPECT_EQ(Slope1Const->getZExtValue(), 0u);
  } else {
    FAIL() << "Slope[1] is not a ConstantInt";
  }

  // Get LinearSeries for Index1
  const LinearSeries *LS1 = Ripple->getLinseriesFor(Index1);
  ASSERT_NE(LS1, nullptr);

  // Verify Index1 LinearSeries: base = 0, slope[0] = 0, slope[1] = 1
  EXPECT_TRUE(LS1->getBaseShape().isScalar());
  EXPECT_EQ(LS1->getSlopeShape().rank(), 2u);
  EXPECT_EQ(LS1->getSlopeShape()[0], 1u);
  EXPECT_EQ(LS1->getSlopeShape()[1], 4u);

  // Base should be constant 0
  if (auto *BaseConst = dyn_cast<ConstantInt>(LS1->getBase())) {
    EXPECT_EQ(BaseConst->getZExtValue(), 0u);
  } else {
    FAIL() << "Base is not a ConstantInt";
  }

  // Slope[0] should be constant 0
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS1->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 0u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Slope[1] should be constant 1
  if (auto *Slope1Const = dyn_cast<ConstantInt>(LS1->getSlope(1))) {
    EXPECT_EQ(Slope1Const->getZExtValue(), 1u);
  } else {
    FAIL() << "Slope[1] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, RippleBlockGetSize) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *GetSizeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_getsize, {IRB.getInt64Ty()});

  // Create a [12][5] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(12);
  Value *Dim1 = IRB.getInt64(5);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  // Get size for dimension 0 -> should be 12
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> GetSize0 =
      IRB.CreateCall(GetSizeFn, {SetShape, Idx0Val});

  // Get size for dimension 1 -> should be 5
  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> GetSize1 =
      IRB.CreateCall(GetSizeFn, {SetShape, Idx1Val});

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Get LinearSeries for GetSize0
  const LinearSeries *LS0 = Ripple->getLinseriesFor(GetSize0);
  ASSERT_NE(LS0, nullptr);

  // Verify GetSize0 LinearSeries: base = 12, all slopes = 0
  EXPECT_TRUE(LS0->getBaseShape().isScalar());
  EXPECT_TRUE(LS0->getSlopeShape().isScalar());

  // Base should be constant 12
  if (auto *BaseConst = dyn_cast<ConstantInt>(LS0->getBase())) {
    EXPECT_EQ(BaseConst->getZExtValue(), 12u);
  } else {
    FAIL() << "Base is not a ConstantInt";
  }
  // All slopes should be constant 0
  EXPECT_TRUE(LS0->hasZeroSlopes());
  for (unsigned i = 0; i < LS0->getSlopeShape().rank(); ++i) {
    if (auto *SlopeConst = dyn_cast<ConstantInt>(LS0->getSlope(i))) {
      EXPECT_EQ(SlopeConst->getZExtValue(), 0u);
    } else {
      FAIL() << "Slope[" << i << "] is not a ConstantInt";
    }
  }

  // Get LinearSeries for GetSize1
  const LinearSeries *LS1 = Ripple->getLinseriesFor(GetSize1);
  ASSERT_NE(LS1, nullptr);

  // Verify GetSize1 LinearSeries: base = 5, all slopes = 0
  EXPECT_TRUE(LS1->getBaseShape().isScalar());
  EXPECT_TRUE(LS1->getSlopeShape().isScalar());

  // Base should be constant 5
  if (auto *BaseConst = dyn_cast<ConstantInt>(LS1->getBase())) {
    EXPECT_EQ(BaseConst->getZExtValue(), 5u);
  } else {
    FAIL() << "Base is not a ConstantInt";
  }

  // All slopes should be constant 0
  EXPECT_TRUE(LS1->hasZeroSlopes());
  for (unsigned i = 0; i < LS1->getSlopeShape().rank(); ++i) {
    if (auto *SlopeConst = dyn_cast<ConstantInt>(LS1->getSlope(i))) {
      EXPECT_EQ(SlopeConst->getZExtValue(), 0u);
    } else {
      FAIL() << "Slope[" << i << "] is not a ConstantInt";
    }
  }
}

TEST_F(PropagateLinearSeriesTest, AddOp) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [10] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(10);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Scalar5 = IRB.getInt64(5);
  Value *Scalar3 = IRB.getInt64(3);
  Value *Scalar2 = IRB.getInt64(2);

  // Test 1: index + scalar -> LS with base = scalar, slope = 1
  AssertingVH<Value> AddIndexScalar =
      IRB.CreateAdd(Index0, Scalar5, "add_index_scalar");

  // Test 2: scalar + index -> LS with base = scalar, slope = 1
  AssertingVH<Value> AddScalarIndex =
      IRB.CreateAdd(Scalar3, Index0, "add_scalar_index");

  // Test 3: (index + scalar1) + scalar2 -> LS with base = scalar1 + scalar2,
  // slope = 1
  AssertingVH<Value> AddChained =
      IRB.CreateAdd(AddIndexScalar, Scalar3, "add_chained");

  // Test 4: ax + b + cx + d -> (a + c)x + (b + d)
  // Index0 * 2. Slope = 2.
  AssertingVH<Value> Mul2 = IRB.CreateMul(Index0, Scalar2, "ls1");
  // Index0 * 3. Slope = 3.
  AssertingVH<Value> Mul3 = IRB.CreateMul(Index0, Scalar3, "ls2");

  // Add: Mul2 + Mul3 -> Slope 5
  AssertingVH<Value> Add = IRB.CreateAdd(Mul2, Mul3, "add_ls1_ls2");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify Index0 is a LinearSeries
  const LinearSeries *LS_Index0 = Ripple->getLinseriesFor(Index0);
  ASSERT_NE(LS_Index0, nullptr);

  // Verify Test 1: index + 5 -> base = 5, slope[0] = 1
  const LinearSeries *LS1 =
      Ripple->getLinseriesFor(cast<Instruction>(&*AddIndexScalar));
  ASSERT_NE(LS1, nullptr);
  EXPECT_TRUE(LS1->getBaseShape().isScalar());

  // Base should be the Add instruction itself
  EXPECT_EQ(LS1->getBase(), AddIndexScalar);

  // Slope[0] should be constant 1
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS1->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 1u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 2: 3 + index -> base = 3, slope[0] = 1
  const LinearSeries *LS2 =
      Ripple->getLinseriesFor(cast<Instruction>(&*AddScalarIndex));
  ASSERT_NE(LS2, nullptr);
  EXPECT_TRUE(LS2->getBaseShape().isScalar());

  // Base should be the Add instruction itself
  EXPECT_EQ(LS2->getBase(), AddScalarIndex);

  // Slope[0] should be constant 1
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS2->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 1u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 3: (index + 5) + 3 -> base = (index + 5) + 3, slope[0] = 1
  const LinearSeries *LS3 =
      Ripple->getLinseriesFor(cast<Instruction>(&*AddChained));
  ASSERT_NE(LS3, nullptr);
  EXPECT_TRUE(LS3->getBaseShape().isScalar());

  // Base should be the chained Add instruction itself
  EXPECT_EQ(LS3->getBase(), AddChained);

  // Slope[0] should be constant 1
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS3->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 1u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify tensor Add
  const LinearSeries *LS_Add =
      Ripple->getLinseriesFor(cast<Instruction>(&*Add));
  ASSERT_NE(LS_Add, nullptr);
  EXPECT_TRUE(LS_Add->getBaseShape().isScalar());
  EXPECT_EQ(LS_Add->getBase(), Add);

  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS_Add->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 5u);
  } else {
    FAIL() << "Add Slope[0] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, SubOp) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [12] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(12);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Scalar20 = IRB.getInt64(20);
  Value *Scalar7 = IRB.getInt64(7);

  // Test 1: scalar - index -> LS with base = scalar, slope = -1
  AssertingVH<Value> SubScalarIndex =
      IRB.CreateSub(Scalar20, Index0, "sub_scalar_index");

  // Test 2: (index + 7) - 3 -> LS with base = 7 - 3 = 4, slope = 1
  AssertingVH<Value> AddFirst = IRB.CreateAdd(Index0, Scalar7, "add_first");
  Value *Scalar3 = IRB.getInt64(3);
  AssertingVH<Value> SubAfterAdd =
      IRB.CreateSub(AddFirst, Scalar3, "sub_after_add");

  Value *Scalar2 = IRB.getInt64(2);

  // Test 4: (ax + b) - (cx + d) -> (a - c)x + (b - d)
  // Index0 * 2. Slope = 2.
  AssertingVH<Value> Mul2 = IRB.CreateMul(Index0, Scalar2, "ls1");
  // Index0 * 3. Slope = 3.
  AssertingVH<Value> Mul3 = IRB.CreateMul(Index0, Scalar3, "ls2");

  // Sub: Mul2 - Mul3 -> Slope -1
  AssertingVH<Value> Sub = IRB.CreateSub(Mul2, Mul3, "sub_ls1_ls2");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify Test 1: scalar - index
  const LinearSeries *LS1 =
      Ripple->getLinseriesFor(cast<Instruction>(&*SubScalarIndex));
  ASSERT_NE(LS1, nullptr);

  // Base should be the Sub instruction itself
  EXPECT_EQ(LS1->getBase(), SubScalarIndex);
  EXPECT_TRUE(LS1->getBaseShape().isScalar());

  // Slope[0] should be constant -1
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS1->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getSExtValue(), -1);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 2: (index + 7) - 3
  const LinearSeries *LS2 =
      Ripple->getLinseriesFor(cast<Instruction>(&*SubAfterAdd));
  ASSERT_NE(LS2, nullptr);

  // Base should be the final Sub instruction
  EXPECT_EQ(LS2->getBase(), SubAfterAdd);
  EXPECT_TRUE(LS2->getBaseShape().isScalar());

  // Slope[0] should be constant 1 (from index)
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS2->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 1u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify tensor Sub
  const LinearSeries *LS_Sub =
      Ripple->getLinseriesFor(cast<Instruction>(&*Sub));
  ASSERT_NE(LS_Sub, nullptr);
  EXPECT_TRUE(LS_Sub->getBaseShape().isScalar());
  EXPECT_EQ(LS_Sub->getBase(), Sub);

  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS_Sub->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getSExtValue(), -1);
  } else {
    FAIL() << "Sub Slope[0] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, MulOp) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [6] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(6);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Scalar2 = IRB.getInt64(2);
  Value *Scalar4 = IRB.getInt64(4);

  // Test 1: index * scalar -> LS with base = 0, slope = scalar
  AssertingVH<Value> MulIndexScalar =
      IRB.CreateMul(Index0, Scalar2, "mul_index_scalar");

  // Test 2: scalar * index -> LS with base = 0, slope = scalar
  AssertingVH<Value> MulScalarIndex =
      IRB.CreateMul(Scalar4, Index0, "mul_scalar_index");

  // Test 3: (index * 2) + 5 -> LS with base = 5, slope = 2
  Value *Scalar5 = IRB.getInt64(5);
  AssertingVH<Value> MulThenAdd =
      IRB.CreateAdd(MulIndexScalar, Scalar5, "mul_then_add");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify Test 1: index * 2 -> base = index * 2, slope[0] = 2
  const LinearSeries *LS1 =
      Ripple->getLinseriesFor(cast<Instruction>(&*MulIndexScalar));
  ASSERT_NE(LS1, nullptr);
  EXPECT_TRUE(LS1->getBaseShape().isScalar());

  // Base should be the Mul instruction itself
  EXPECT_EQ(LS1->getBase(), MulIndexScalar);

  // Slope[0] should be constant 2
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS1->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 2u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 2: 4 * index -> base = 4 * index, slope[0] = 4
  const LinearSeries *LS2 =
      Ripple->getLinseriesFor(cast<Instruction>(&*MulScalarIndex));
  ASSERT_NE(LS2, nullptr);
  EXPECT_TRUE(LS2->getBaseShape().isScalar());

  // Base should be the Mul instruction itself
  EXPECT_EQ(LS2->getBase(), MulScalarIndex);

  // Slope[0] should be constant 4
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS2->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 4u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 3: (index * 2) + 5 -> base = (index * 2) + 5, slope[0] = 2
  const LinearSeries *LS3 =
      Ripple->getLinseriesFor(cast<Instruction>(&*MulThenAdd));
  ASSERT_NE(LS3, nullptr);
  EXPECT_TRUE(LS3->getBaseShape().isScalar());

  // Base should be the final Add instruction
  EXPECT_EQ(LS3->getBase(), MulThenAdd);

  // Slope[0] should be constant 2 (from the multiplication)
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS3->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 2u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, GEP) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [16] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(16);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  // Create a base pointer (array of i64)
  Type *Int64Ty = IRB.getInt64Ty();
  Type *ArrayTy = ArrayType::get(Int64Ty, 100);
  AssertingVH<Value> BasePtr = IRB.CreateAlloca(ArrayTy, nullptr, "base_ptr");

  // Test 1: GEP with scalar base and tensor index
  // base_ptr[index] -> LS with base = base_ptr, slope = sizeof(i64)
  AssertingVH<Value> GEP1 = IRB.CreateGEP(
      ArrayTy, BasePtr, {IRB.getInt64(0), static_cast<Value *>(Index0)},
      "gep_scalar_base_tensor_index");

  // Test 2: GEP with tensor base (from previous GEP) and scalar index
  Value *ScalarIdx = IRB.getInt64(2);
  AssertingVH<Value> GEP2 =
      IRB.CreateGEP(Int64Ty, GEP1, ScalarIdx, "gep_tensor_base_scalar_index");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify Test 1: GEP with scalar base and tensor index
  const LinearSeries *LS1 = Ripple->getLinseriesFor(cast<Instruction>(&*GEP1));
  ASSERT_NE(LS1, nullptr);

  // Base should be the GEP instruction itself
  EXPECT_EQ(LS1->getBase(), GEP1);
  EXPECT_TRUE(LS1->getBaseShape().isScalar());

  // Slope[0] should be sizeof(i64) in bytes
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS1->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(),
              F->getDataLayout().getTypeAllocSize(Int64Ty));
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 2: GEP with tensor base and scalar index
  const LinearSeries *LS2 = Ripple->getLinseriesFor(cast<Instruction>(&*GEP2));
  ASSERT_NE(LS2, nullptr);

  // Base should be the GEP instruction itself
  EXPECT_EQ(LS2->getBase(), GEP2);
  EXPECT_TRUE(LS2->getBaseShape().isScalar());

  // Slope[0] should be constant sizeof(i64) (inherited from GEP1)
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS2->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(),
              F->getDataLayout().getTypeAllocSize(Int64Ty));
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, GEPStructArray2D) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [8][4] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(8);
  Value *Dim1 = IRB.getInt64(4);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 = IRB.CreateCall(IndexFn, {SetShape, Idx1Val});

  // Struct: { i1, [8 x float] }
  Type *FloatTy = IRB.getFloatTy();
  ArrayType *FloatArrayTy = ArrayType::get(FloatTy, 8);
  StructType *StructTy = StructType::get(C, {IRB.getInt1Ty(), FloatArrayTy});

  // Array of structs: [20 x Struct]
  ArrayType *OuterArrTy = ArrayType::get(StructTy, 20);
  AssertingVH<Value> BasePtr =
      IRB.CreateAlloca(OuterArrTy, nullptr, "base_ptr");

  // GEP: array[Index1].field1[Index0]
  // Indices: 0, Index1, 1, Index0
  AssertingVH<Value> GEP =
      IRB.CreateGEP(OuterArrTy, BasePtr,
                    {IRB.getInt64(0), static_cast<Value *>(Index1),
                     IRB.getInt32(1), static_cast<Value *>(Index0)},
                    "gep_2d");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  const LinearSeries *LS = Ripple->getLinseriesFor(cast<Instruction>(&*GEP));
  ASSERT_NE(LS, nullptr);

  // Base should be the GEP instruction itself
  EXPECT_EQ(LS->getBase(), GEP);
  EXPECT_TRUE(LS->getBaseShape().isScalar());

  // Slope[0] corresponds to Index0 -> should be sizeof(float)
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(),
              F->getDataLayout().getTypeAllocSize(FloatTy));
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Slope[1] corresponds to Index1 -> should be sizeof(struct)
  if (auto *Slope1Const = dyn_cast<ConstantInt>(LS->getSlope(1))) {
    EXPECT_EQ(Slope1Const->getZExtValue(),
              F->getDataLayout().getTypeAllocSize(StructTy));
  } else {
    FAIL() << "Slope[1] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, GEP2DArraySharedIndex) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [10] shape (1D)
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(10);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  // Type: [10 x [20 x float]]
  // Inner array: [20 x float]
  Type *FloatTy = IRB.getFloatTy();
  ArrayType *InnerArrTy = ArrayType::get(FloatTy, 20);
  ArrayType *OuterArrTy = ArrayType::get(InnerArrTy, 10);

  AssertingVH<Value> BasePtr =
      IRB.CreateAlloca(OuterArrTy, nullptr, "base_ptr");

  // GEP: array[Index0][Index0]
  // Indices: 0, Index0, Index0
  AssertingVH<Value> GEP =
      IRB.CreateGEP(OuterArrTy, BasePtr,
                    {IRB.getInt64(0), static_cast<Value *>(Index0),
                     static_cast<Value *>(Index0)},
                    "gep_shared_index");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  const LinearSeries *LS = Ripple->getLinseriesFor(cast<Instruction>(&*GEP));
  ASSERT_NE(LS, nullptr);

  // Base should be the GEP instruction itself
  EXPECT_EQ(LS->getBase(), GEP);
  EXPECT_TRUE(LS->getBaseShape().isScalar());

  // Slope[0] should be sizeof(InnerArrTy) + sizeof(float)
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS->getSlope(0))) {
    uint64_t ExpectedSlope = F->getDataLayout().getTypeAllocSize(InnerArrTy) +
                             F->getDataLayout().getTypeAllocSize(FloatTy);
    EXPECT_EQ(Slope0Const->getZExtValue(), ExpectedSlope);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, PhiInductionVariable) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);

  // Create basic blocks: Entry -> LoopHeader -> LoopBody -> LoopLatch -> Exit
  BasicBlock *EntryBB = BasicBlock::Create(C, "Entry", F);
  BasicBlock *LoopHeaderBB = BasicBlock::Create(C, "LoopHeader", F);
  BasicBlock *LoopBodyBB = BasicBlock::Create(C, "LoopBody", F);
  BasicBlock *LoopLatchBB = BasicBlock::Create(C, "LoopLatch", F);
  BasicBlock *ExitBB = BasicBlock::Create(C, "Exit", F);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Entry Block: Set up initial values
  IRB.SetInsertPoint(EntryBB);
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(20);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> RippleId = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *InitCounter = IRB.getInt64(0);
  Value *LoopBound = IRB.getInt64(10);

  IRB.CreateBr(LoopHeaderBB);

  // Loop Header: PHI for induction variable
  IRB.SetInsertPoint(LoopHeaderBB);
  AssertingVH<PHINode> InductionVar =
      IRB.CreatePHI(IRB.getInt64Ty(), 2, "induction_var");
  InductionVar->addIncoming(InitCounter, EntryBB);

  // Loop condition
  AssertingVH<Value> LoopCond =
      IRB.CreateICmpULT(InductionVar, LoopBound, "loop_cond");
  IRB.CreateCondBr(LoopCond, LoopBodyBB, ExitBB);

  // Loop Body: Use induction variable
  IRB.SetInsertPoint(LoopBodyBB);
  AssertingVH<Value> BodyOp = IRB.CreateAdd(InductionVar, RippleId, "body_op");
  IRB.CreateBr(LoopLatchBB);

  // Loop Latch: Increment induction variable
  IRB.SetInsertPoint(LoopLatchBB);
  AssertingVH<Value> NextCounter =
      IRB.CreateAdd(InductionVar, One, "next_counter");
  InductionVar->addIncoming(NextCounter, LoopLatchBB);
  IRB.CreateBr(LoopHeaderBB);

  // Exit Block
  IRB.SetInsertPoint(ExitBB);
  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify InductionVar is recognized as a LinearSeries
  const LinearSeries *LS_Induction =
      Ripple->getLinseriesFor(cast<Instruction>(&*InductionVar));
  ASSERT_NE(LS_Induction, nullptr);

  // Base should be the PHI node itself
  EXPECT_EQ(LS_Induction->getBase(), InductionVar);
  EXPECT_TRUE(LS_Induction->getBaseShape().isScalar());

  // The LinearSeries should be scalar
  EXPECT_TRUE(LS_Induction->isScalar());

  // Verify NextCounter is also a LinearSeries
  const LinearSeries *LS_Next =
      Ripple->getLinseriesFor(cast<Instruction>(&*NextCounter));
  ASSERT_NE(LS_Next, nullptr);

  // Base should be the Add instruction itself
  EXPECT_EQ(LS_Next->getBase(), NextCounter);
  EXPECT_TRUE(LS_Next->getBaseShape().isScalar());
}

TEST_F(PropagateLinearSeriesTest, PhiDifferentSlopes) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);

  // Create blocks
  BasicBlock *EntryBB = BasicBlock::Create(C, "Entry", F);
  BasicBlock *IfBB = BasicBlock::Create(C, "If", F);
  BasicBlock *ElseBB = BasicBlock::Create(C, "Else", F);
  BasicBlock *MergeBB = BasicBlock::Create(C, "Merge", F);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Entry
  IRB.SetInsertPoint(EntryBB);
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(10);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> RippleId = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  // Condition
  Value *Cond = IRB.CreateICmpEQ(IRB.getInt64(3), IRB.getInt64(5));
  IRB.CreateCondBr(Cond, IfBB, ElseBB);

  // If Branch
  IRB.SetInsertPoint(IfBB);
  Value *ValIf = IRB.CreateMul(RippleId, IRB.getInt64(2), "val_if");
  IRB.CreateBr(MergeBB);

  // Else Branch
  IRB.SetInsertPoint(ElseBB);
  Value *ValElse = IRB.CreateMul(RippleId, IRB.getInt64(4), "val_else");
  IRB.CreateBr(MergeBB);

  // Merge Branch
  IRB.SetInsertPoint(MergeBB);
  PHINode *PhiVal = IRB.CreatePHI(IRB.getInt64Ty(), 2, "phi_val");
  PhiVal->addIncoming(ValIf, IfBB);
  PhiVal->addIncoming(ValElse, ElseBB);

  Value *Result = IRB.CreateAdd(PhiVal, RippleId, "result");
  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  const LinearSeries *LS_ValIf =
      Ripple->getLinseriesFor(cast<Instruction>(ValIf));
  ASSERT_NE(LS_ValIf, nullptr);
  const LinearSeries *LS_ValElse =
      Ripple->getLinseriesFor(cast<Instruction>(ValElse));
  ASSERT_NE(LS_ValElse, nullptr);
  EXPECT_EQ(LS_ValIf->getBaseShape(), LS_ValIf->getBaseShape());
  EXPECT_EQ(LS_ValIf->getSlopeShape(), LS_ValIf->getSlopeShape());
  EXPECT_FALSE(LS_ValIf->getSlopeShape().isScalar());

  // Verification for PhiVal
  const LinearSeries *LS_Phi =
      Ripple->getLinseriesFor(cast<Instruction>(PhiVal));
  ASSERT_NE(LS_Phi, nullptr);
  EXPECT_TRUE(LS_Phi->getBaseShape().isScalar());
  EXPECT_FALSE(LS_Phi->getSlopeShape().isScalar());

  Value *PhiSlopeVal = LS_Phi->getSlope(0);
  ASSERT_NE(PhiSlopeVal, nullptr);

  PHINode *PhiSlope = dyn_cast<PHINode>(PhiSlopeVal);
  ASSERT_TRUE(PhiSlope) << "PhiVal Slope is not a PHI node";
  EXPECT_EQ(PhiSlope->getNumIncomingValues(), 2u);

  // We can't guarantee order, so check existence of 2 and 4 in PhiSlope
  bool PhiFound2 = false;
  bool PhiFound4 = false;

  for (Value *Inc : PhiSlope->incoming_values()) {
    if (ConstantInt *C = dyn_cast<ConstantInt>(Inc)) {
      if (C->getZExtValue() == 2)
        PhiFound2 = true;
      if (C->getZExtValue() == 4)
        PhiFound4 = true;
    }
  }

  EXPECT_TRUE(PhiFound2);
  EXPECT_TRUE(PhiFound4);

  // Verification for Result
  const LinearSeries *LS = Ripple->getLinseriesFor(cast<Instruction>(Result));
  ASSERT_NE(LS, nullptr);

  // Verify Slope
  Value *Slope = LS->getSlope(0);
  ASSERT_NE(Slope, nullptr);

  // Expect Slope to be an Add instruction (1 + PHI)
  Instruction *SlopeInst = dyn_cast<Instruction>(Slope);
  ASSERT_TRUE(SlopeInst) << "Slope is not an instruction";
  EXPECT_EQ(SlopeInst->getOpcode(), Instruction::Add);

  // Check operands
  Value *Op0 = SlopeInst->getOperand(0);
  Value *Op1 = SlopeInst->getOperand(1);

  ConstantInt *C0 = dyn_cast<ConstantInt>(Op0);
  ConstantInt *C1 = dyn_cast<ConstantInt>(Op1);
  PHINode *ResPhiSlope = nullptr;

  if (C0 && C0->isOne()) {
    ResPhiSlope = dyn_cast<PHINode>(Op1);
  } else if (C1 && C1->isOne()) {
    ResPhiSlope = dyn_cast<PHINode>(Op0);
  }

  ASSERT_TRUE(ResPhiSlope) << "Slope Add does not contain 1 and PHI";

  // Check PHI incoming values (2 and 4)
  ASSERT_EQ(ResPhiSlope->getNumIncomingValues(), 2u);

  // We can't guarantee order, so check existence
  bool Found2 = false;
  bool Found4 = false;

  for (Value *Inc : ResPhiSlope->incoming_values()) {
    if (ConstantInt *C = dyn_cast<ConstantInt>(Inc)) {
      if (C->getZExtValue() == 2)
        Found2 = true;
      if (C->getZExtValue() == 4)
        Found4 = true;
    }
  }

  EXPECT_TRUE(Found2);
  EXPECT_TRUE(Found4);
}

TEST_F(PropagateLinearSeriesTest, PhiDontPropagateOutsideVectorBranches) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);

  // Create blocks
  BasicBlock *EntryBB = BasicBlock::Create(C, "Entry", F);
  BasicBlock *IfBB = BasicBlock::Create(C, "If", F);
  BasicBlock *ElseBB = BasicBlock::Create(C, "Else", F);
  BasicBlock *MergeBB = BasicBlock::Create(C, "Merge", F);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Entry
  IRB.SetInsertPoint(EntryBB);
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(10);
  Value *Dim1 = IRB.getInt64(20);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> RippleId = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  AssertingVH<CallInst> RippleId2 = IRB.CreateCall(IndexFn, {SetShape, One});

  // TensorBase [10] (forced base via Select)
  AssertingVH<Value> TensorBase =
      IRB.CreateSelect(IRB.getInt1(false), RippleId, RippleId, "tensor_base");

  // Vector Condition
  Value *Cond = IRB.CreateICmpEQ(RippleId, IRB.getInt64(5));
  IRB.CreateCondBr(Cond, IfBB, ElseBB);

  // If Branch
  IRB.SetInsertPoint(IfBB);
  Value *ValIf = IRB.CreateMul(RippleId, IRB.getInt64(2), "val_if");
  Value *ValIf2 = IRB.CreateAdd(TensorBase, RippleId2, "val_if");
  IRB.CreateBr(MergeBB);

  // Else Branch
  IRB.SetInsertPoint(ElseBB);
  Value *ValElse = IRB.CreateMul(RippleId, IRB.getInt64(4), "val_else");
  Value *ValElse2 = IRB.CreateSub(TensorBase, RippleId2, "val_else");
  IRB.CreateBr(MergeBB);

  // Merge Branch
  IRB.SetInsertPoint(MergeBB);
  PHINode *PhiVal = IRB.CreatePHI(IRB.getInt64Ty(), 2, "phi_val");
  PhiVal->addIncoming(ValIf, IfBB);
  PhiVal->addIncoming(ValElse, ElseBB);
  PHINode *PhiVal2 = IRB.CreatePHI(IRB.getInt64Ty(), 2, "phi_val_2");
  PhiVal2->addIncoming(ValIf2, IfBB);
  PhiVal2->addIncoming(ValElse2, ElseBB);

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verification that the Phi restarts a linear series
  const LinearSeries *LS_Phi =
      Ripple->getLinseriesFor(cast<Instruction>(PhiVal));
  ASSERT_NE(LS_Phi, nullptr);
  EXPECT_FALSE(LS_Phi->getBaseShape().isScalar());
  EXPECT_TRUE(LS_Phi->getSlopeShape().isScalar());
  EXPECT_TRUE(LS_Phi->hasZeroSlopes());

  const LinearSeries *LS_Tensor =
      Ripple->getLinseriesFor(cast<Instruction>(&*TensorBase));
  ASSERT_NE(LS_Tensor, nullptr);

  const LinearSeries *LS_Phi2 =
      Ripple->getLinseriesFor(cast<Instruction>(PhiVal2));
  ASSERT_NE(LS_Phi2, nullptr);
  // TODO: change this when we have better support for the case where
  // LHS.baseShape() != RHS.getBaseShape() in LS propagation!
  // We can optimize to a base of shape [10] and slope [1][20]
  // EXPECT_EQ(LS_Phi2->getBaseShape(), LS_Tensor->getBaseShape());
  // EXPECT_FALSE(LS_Phi2->getSlopeShape().isScalar());
  EXPECT_NE(LS_Phi2->getBaseShape(), LS_Tensor->getBaseShape());
  EXPECT_TRUE(LS_Phi2->getSlopeShape().isScalar());
}

TEST_F(PropagateLinearSeriesTest, CastOp) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [8] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(8);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Scalar10 = IRB.getInt64(10);

  // Create a LinearSeries: index * 2 + 10
  AssertingVH<Value> MulIndex = IRB.CreateMul(Index0, IRB.getInt64(2), "mul");
  AssertingVH<Value> AddScalar =
      IRB.CreateAdd(MulIndex, Scalar10, "add", /*NUW*/ true, /*NSW*/ true);

  // Test 1: Cast i64 to i32 (Trunc NUW) -> LS should propagate
  AssertingVH<Value> TruncCastNUW =
      IRB.CreateTrunc(AddScalar, IRB.getInt32Ty(), "trunc_cast_nuw",
                      /*NUW*/ true, /*NSW*/ false);

  // Test 2: Cast i64 to i32 (Trunc NSW) -> LS should propagate
  AssertingVH<Value> TruncCastNSW =
      IRB.CreateTrunc(AddScalar, IRB.getInt32Ty(), "trunc_cast_nsw",
                      /*NUW*/ false, /*NSW*/ true);

  // Test 3: Cast i64 to i32 (Trunc plain) -> LS should NOT propagate
  AssertingVH<Value> TruncCastPlain =
      IRB.CreateTrunc(AddScalar, IRB.getInt32Ty(), "trunc_cast_plain");
  // Add a user that does not have no-wrap semantics to prevent propagation
  IRB.CreateAdd(TruncCastPlain, IRB.getInt32(1), "prevent_propagation");

  // Test 4: Cast i64 to i128 (ZExt) -> LS should propagate
  AssertingVH<Value> ExtCast =
      IRB.CreateZExt(AddScalar, IRB.getInt128Ty(), "ext_cast");

  // Test 5: Cast i64 to i128 (SExt) -> LS should propagate
  AssertingVH<Value> SExtCast =
      IRB.CreateSExt(AddScalar, IRB.getInt128Ty(), "sext_cast");

  // Test 6: BitCast -> LS should NOT propagate (using double as it has same
  // size 64)
  AssertingVH<Value> BitCast =
      IRB.CreateBitCast(AddScalar, IRB.getDoubleTy(), "bit_cast");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  const LinearSeries *LS_MulIndex =
      Ripple->getLinseriesFor(cast<Instruction>(&*MulIndex));
  ASSERT_NE(LS_MulIndex, nullptr);
  EXPECT_EQ(LS_MulIndex->getBase(), MulIndex);
  EXPECT_TRUE(LS_MulIndex->getBaseShape().isScalar());

  const LinearSeries *LS_AddScalar =
      Ripple->getLinseriesFor(cast<Instruction>(&*AddScalar));
  ASSERT_NE(LS_AddScalar, nullptr);
  EXPECT_EQ(LS_AddScalar->getBase(), AddScalar);
  EXPECT_TRUE(LS_AddScalar->getBaseShape().isScalar());

  // Verify Test 1: Trunc NUW cast propagates LinearSeries
  const LinearSeries *LS_TruncNUW =
      Ripple->getLinseriesFor(cast<Instruction>(&*TruncCastNUW));
  ASSERT_NE(LS_TruncNUW, nullptr);
  EXPECT_EQ(LS_TruncNUW->getBase(), TruncCastNUW);
  EXPECT_TRUE(LS_TruncNUW->getBaseShape().isScalar());
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS_TruncNUW->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 2u);
    EXPECT_EQ(Slope0Const->getType()->getIntegerBitWidth(), 32u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 2: Trunc NSW cast propagates LinearSeries
  const LinearSeries *LS_TruncNSW =
      Ripple->getLinseriesFor(cast<Instruction>(&*TruncCastNSW));
  ASSERT_NE(LS_TruncNSW, nullptr);
  EXPECT_EQ(LS_TruncNSW->getBase(), TruncCastNSW);
  EXPECT_TRUE(LS_TruncNSW->getBaseShape().isScalar());
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS_TruncNSW->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 2u);
    EXPECT_EQ(Slope0Const->getType()->getIntegerBitWidth(), 32u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 3: Trunc plain cast DOES NOT propagate LinearSeries
  const LinearSeries *LS_TruncPlain =
      Ripple->getLinseriesFor(cast<Instruction>(&*TruncCastPlain));
  // Expecting a new linear series with the tensor base and no slopes
  ASSERT_NE(LS_TruncPlain, nullptr);
  EXPECT_FALSE(LS_TruncPlain->getBaseShape().isScalar());
  EXPECT_TRUE(LS_TruncPlain->getSlopeShape().isScalar());
  EXPECT_TRUE(LS_TruncPlain->hasZeroSlopes());

  // Verify Test 4: Ext cast propagates LinearSeries
  const LinearSeries *LS_Ext =
      Ripple->getLinseriesFor(cast<Instruction>(&*ExtCast));
  ASSERT_NE(LS_Ext, nullptr);
  EXPECT_EQ(LS_Ext->getBase(), ExtCast);
  EXPECT_TRUE(LS_Ext->getBaseShape().isScalar());
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS_Ext->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 2u);
    EXPECT_EQ(Slope0Const->getType()->getIntegerBitWidth(), 128u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 5: SExt cast propagates LinearSeries
  const LinearSeries *LS_SExt =
      Ripple->getLinseriesFor(cast<Instruction>(&*SExtCast));
  ASSERT_NE(LS_SExt, nullptr);
  EXPECT_EQ(LS_SExt->getBase(), SExtCast);
  EXPECT_TRUE(LS_SExt->getBaseShape().isScalar());
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS_SExt->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 2u);
    EXPECT_EQ(Slope0Const->getType()->getIntegerBitWidth(), 128u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 6: BitCast DOES NOT propagate LinearSeries
  const LinearSeries *LS_BitCast =
      Ripple->getLinseriesFor(cast<Instruction>(&*BitCast));
  ASSERT_EQ(LS_BitCast, nullptr);
}

TEST_F(PropagateLinearSeriesTest, ShlOp) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [8] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(8);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Scalar2 = IRB.getInt64(2);

  // Test: index << 2 -> LS with base = index << 2, slope = 1 << 2 = 4
  AssertingVH<Value> ShlIndex = IRB.CreateShl(Index0, Scalar2, "shl_index");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify: index << 2
  const LinearSeries *LS_Shl =
      Ripple->getLinseriesFor(cast<Instruction>(&*ShlIndex));
  ASSERT_NE(LS_Shl, nullptr);

  // Base should be the Shl instruction itself
  EXPECT_EQ(LS_Shl->getBase(), ShlIndex);
  EXPECT_TRUE(LS_Shl->getBaseShape().isScalar());

  // Slope[0] should be constant 4 (1 << 2)
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS_Shl->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 4u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, RippleBroadcast) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});
  Function *BroadcastFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_broadcast, {IRB.getInt64Ty()});

  // Create a [6][4] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(6);
  Value *Dim1 = IRB.getInt64(4);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Scalar7 = IRB.getInt64(7);

  // Test 1: Broadcast scalar to [6] (bitset 0x1 = dimension 0)
  Value *Bitset1D = IRB.getInt64(0x1);
  AssertingVH<CallInst> Broadcast1 = IRB.CreateCall(
      BroadcastFn, {SetShape, Bitset1D, Scalar7}, "broadcast_scalar_to_1d");

  // Test 2: Broadcast index to [6][4] (bitset 0x2 = dimension 1)
  Value *BitsetDim1 = IRB.getInt64(0x2);
  AssertingVH<CallInst> Broadcast2 = IRB.CreateCall(
      BroadcastFn, {SetShape, BitsetDim1, Index0}, "broadcast_index_to_2d");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify Test 1: Broadcast scalar to [6]
  const LinearSeries *LS1 =
      Ripple->getLinseriesFor(cast<Instruction>(&*Broadcast1));
  ASSERT_NE(LS1, nullptr);

  // Base should be scalar 7
  EXPECT_TRUE(LS1->getBaseShape().isScalar());
  if (auto *BaseConst = dyn_cast<ConstantInt>(LS1->getBase())) {
    EXPECT_EQ(BaseConst->getZExtValue(), 7u);
  }

  // Slope[0] should be 0 (broadcasting)
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS1->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 0u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Verify Test 2: Broadcast index [6] to [6][4]
  const LinearSeries *LS_Index0 =
      Ripple->getLinseriesFor(cast<Instruction>(&*Index0));
  ASSERT_NE(LS_Index0, nullptr);

  const LinearSeries *LS2 =
      Ripple->getLinseriesFor(cast<Instruction>(&*Broadcast2));
  ASSERT_NE(LS2, nullptr);

  // Base should be 0 (it's the ripple.index base)
  EXPECT_EQ(LS2->getBase(), LS_Index0->getBase());
  EXPECT_TRUE(LS2->getBaseShape().isScalar());
  if (auto *BaseConst = dyn_cast<ConstantInt>(LS2->getBase())) {
    EXPECT_EQ(BaseConst->getZExtValue(), 0u);
  } else {
    FAIL() << "Base is not a ConstantInt";
  }

  // Slope[0] should be 1 (from Index0)
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS2->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 1u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Slope[1] should be 0 (broadcasting along dimension 1)
  if (auto *Slope1Const = dyn_cast<ConstantInt>(LS2->getSlope(1))) {
    EXPECT_EQ(Slope1Const->getZExtValue(), 0u);
  } else {
    FAIL() << "Slope[1] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, BroadcastThroughCast) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *BroadcastFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_broadcast, {IRB.getInt64Ty()});

  // Create a [8] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(8);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Scalar42 = IRB.getInt64(42);
  Value *Bitset1D = IRB.getInt64(0x1); // Broadcast to dim 0

  // Broadcast scalar to [8]
  AssertingVH<CallInst> Broadcast = IRB.CreateCall(
      BroadcastFn, {SetShape, Bitset1D, Scalar42}, "broadcast_scalar");

  // Cast i64 to i32 (Trunc plain - NO NUW/NSW)
  // This would fail to propagate if it wasn't a broadcast/scalar series
  AssertingVH<Value> TruncCast =
      IRB.CreateTrunc(Broadcast, IRB.getInt32Ty(), "trunc_cast");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify TruncCast propagates LinearSeries despite being a plain Trunc
  const LinearSeries *LS_Trunc =
      Ripple->getLinseriesFor(cast<Instruction>(&*TruncCast));
  ASSERT_NE(LS_Trunc, nullptr);

  // Base should be the Trunc instruction itself (as per implementation)
  EXPECT_EQ(LS_Trunc->getBase(), TruncCast);
  EXPECT_TRUE(LS_Trunc->getBaseShape().isScalar());

  // Slope should be 0 (preserved from broadcast)
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS_Trunc->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 0u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }
}

TEST_F(PropagateLinearSeriesTest, BroadcastThroughCastTensor) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});
  Function *BroadcastFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_broadcast, {IRB.getInt64Ty()});

  // Create a [4][8] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(4);
  Value *Dim1 = IRB.getInt64(8);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  // Create Index for dimension 0 -> [4]
  // This has a LinearSeries with Slope 1, so it's not "ZeroSlopes".
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  // Create a Tensor Base with Zero Slopes by using an instruction not handled
  // by LinearSeries construction (Select), forcing it to become a Base.
  // Select(False, Index0, Index0) -> Shape [4]
  Value *False = IRB.getInt1(false);
  AssertingVH<Value> TensorBase =
      IRB.CreateSelect(False, Index0, Index0, "tensor_base");

  // Broadcast [4] to [4][8] (along dimension 1)
  Value *BitsetDim1 = IRB.getInt64(0x2);
  AssertingVH<CallInst> Broadcast = IRB.CreateCall(
      BroadcastFn, {SetShape, BitsetDim1, TensorBase}, "broadcast_tensor");

  // Cast i64 to i32 (Trunc plain - NO NUW/NSW)
  AssertingVH<Value> TruncCast =
      IRB.CreateTrunc(Broadcast, IRB.getInt32Ty(), "trunc_cast");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify TruncCast propagates LinearSeries
  const LinearSeries *LS_Trunc =
      Ripple->getLinseriesFor(dyn_cast<Instruction>(&*TruncCast));
  ASSERT_NE(LS_Trunc, nullptr);

  // Base should be the Trunc instruction itself
  EXPECT_EQ(LS_Trunc->getBase(), TruncCast);

  // Base Shape should be [4] (from TensorBase)
  EXPECT_FALSE(LS_Trunc->getBaseShape().isScalar());
  EXPECT_EQ(LS_Trunc->getBaseShape()[0], 4u);
  EXPECT_EQ(LS_Trunc->getBaseShape()[1], 1u);

  // Slopes should be 0
  EXPECT_TRUE(LS_Trunc->hasZeroSlopes());

  // Slope Shape should be [1][8] (forming a Broadcast)
  EXPECT_EQ(LS_Trunc->getSlopeShape().rank(), 2u);
  EXPECT_EQ(LS_Trunc->getSlopeShape()[0], 1u);
  EXPECT_EQ(LS_Trunc->getSlopeShape()[1], 8u);
}

TEST_F(PropagateLinearSeriesTest, BroadcastPropagationScalarBinop) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [10] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(10);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Zero = IRB.getInt64(0);

  // 1. Create a Splat (Scalar Base, Zero Slope) via Index * 0
  AssertingVH<Value> Splat = IRB.CreateMul(Index0, Zero, "splat");

  // 2. UnaryOp on Splat -> Should propagate Splat (Scalar Base)
  AssertingVH<Value> NegSplat = IRB.CreateNeg(Splat, "neg_splat");

  // 3. Mul Splat * Splat -> Should propagate Splat (Scalar Base)
  AssertingVH<Value> MulSplat = IRB.CreateMul(Splat, Splat, "mul_splat");

  // 4. Create a TensorBase (Vector Base, Zero Slope) via Index * Index
  AssertingVH<Value> TensorBase = IRB.CreateMul(Index0, Index0, "tensor_base");

  // 5. UnaryOp on TensorBase -> Should propagate TensorBase (Vector Base)
  AssertingVH<Value> NegTensor = IRB.CreateNeg(TensorBase, "neg_tensor");

  // 6. Mul TensorBase * TensorBase -> Should propagate TensorBase (Vector Base)
  AssertingVH<Value> MulTensor =
      IRB.CreateMul(TensorBase, TensorBase, "mul_tensor");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify Splat (Base check)
  const LinearSeries *LS_Splat =
      Ripple->getLinseriesFor(cast<Instruction>(&*Splat));
  ASSERT_NE(LS_Splat, nullptr);
  EXPECT_TRUE(LS_Splat->getBaseShape().isScalar());
  EXPECT_TRUE(LS_Splat->hasZeroSlopes());
  EXPECT_TRUE(LS_Splat->getShape().isVector());

  // Verify NegSplat (Propagation check)
  const LinearSeries *LS_NegSplat =
      Ripple->getLinseriesFor(cast<Instruction>(&*NegSplat));
  ASSERT_NE(LS_NegSplat, nullptr);
  // Should preserve Scalar Base (Splat nature)
  EXPECT_TRUE(LS_NegSplat->getBaseShape().isScalar());
  EXPECT_TRUE(LS_NegSplat->hasZeroSlopes());
  EXPECT_TRUE(LS_NegSplat->getShape().isVector());

  // Verify MulSplat (Propagation check)
  const LinearSeries *LS_MulSplat =
      Ripple->getLinseriesFor(cast<Instruction>(&*MulSplat));
  ASSERT_NE(LS_MulSplat, nullptr);
  // Should preserve Scalar Base
  EXPECT_TRUE(LS_MulSplat->getBaseShape().isScalar());
  EXPECT_TRUE(LS_MulSplat->hasZeroSlopes());
  EXPECT_TRUE(LS_MulSplat->getShape().isVector());

  // Verify TensorBase (Base check)
  const LinearSeries *LS_Tensor =
      Ripple->getLinseriesFor(cast<Instruction>(&*TensorBase));
  ASSERT_NE(LS_Tensor, nullptr);
  EXPECT_TRUE(LS_Tensor->getBaseShape().isVector());
  EXPECT_TRUE(LS_Tensor->hasZeroSlopes());
  EXPECT_TRUE(LS_Tensor->getSlopeShape().isScalar());

  // Verify NegTensor
  const LinearSeries *LS_NegTensor =
      Ripple->getLinseriesFor(cast<Instruction>(&*NegTensor));
  ASSERT_NE(LS_NegTensor, nullptr);
  EXPECT_TRUE(LS_NegTensor->getBaseShape().isVector());
  EXPECT_TRUE(LS_NegTensor->hasZeroSlopes());
  EXPECT_TRUE(LS_NegTensor->getSlopeShape().isScalar());

  // Verify MulTensor
  const LinearSeries *LS_MulTensor =
      Ripple->getLinseriesFor(cast<Instruction>(&*MulTensor));
  ASSERT_NE(LS_MulTensor, nullptr);
  EXPECT_TRUE(LS_MulTensor->getBaseShape().isVector());
  EXPECT_TRUE(LS_MulTensor->hasZeroSlopes());
  EXPECT_TRUE(LS_MulTensor->getSlopeShape().isScalar());
}

TEST_F(PropagateLinearSeriesTest, BroadcastPropagationTensor) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});
  Function *BroadcastFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_broadcast, {IRB.getInt64Ty()});

  // Create a [4][8][16] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(4);
  Value *Dim1 = IRB.getInt64(8);
  Value *Dim2 = IRB.getInt64(16);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, Dim2, One, One, One, One, One, One, One});

  // Index 0 -> [4]
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  // TensorBase [4] (forced base via Select)
  Value *False = IRB.getInt1(false);
  AssertingVH<Value> TensorBase =
      IRB.CreateSelect(False, Index0, Index0, "tensor_base");

  // ScalarBase
  Value *ScalarBase = IRB.getInt64(42);

  // --- Scalar Base with Non-Scalar Zero Slope ---
  // Broadcast scalar to [4][1][1] (dim 0)
  Value *BitsetDim0 = IRB.getInt64(0x1);
  AssertingVH<CallInst> BroadcastScalar1 = IRB.CreateCall(
      BroadcastFn, {SetShape, BitsetDim0, ScalarBase}, "bcast_scalar_dim0");

  // Broadcast scalar to [1][8][1] (dim 1)
  Value *BitsetDim1 = IRB.getInt64(0x2);
  AssertingVH<CallInst> BroadcastScalar2 = IRB.CreateCall(
      BroadcastFn, {SetShape, BitsetDim1, ScalarBase}, "bcast_scalar_dim1");

  // Add: [4][1][1] + [1][8][1] -> [4][8][1]
  AssertingVH<Value> AddScalar =
      IRB.CreateAdd(BroadcastScalar1, BroadcastScalar2, "add_scalar");

  // --- Tensor Base with Non-Scalar Zero Slope (3D) ---
  // TensorBase is [4, 1, 1] effectively

  // Broadcast TensorBase to [4][8][1] (dim 1)
  AssertingVH<CallInst> BroadcastTensor1 = IRB.CreateCall(
      BroadcastFn, {SetShape, BitsetDim1, TensorBase}, "bcast_tensor_dim1");

  // Broadcast TensorBase to [4][1][16] (dim 2)
  Value *BitsetDim2 = IRB.getInt64(0x4);
  AssertingVH<CallInst> BroadcastTensor2 = IRB.CreateCall(
      BroadcastFn, {SetShape, BitsetDim2, TensorBase}, "bcast_tensor_dim2");

  // Add: [4][8][1] + [4][1][16] -> [4][8][16]
  AssertingVH<Value> AddTensor =
      IRB.CreateAdd(BroadcastTensor1, BroadcastTensor2, "add_tensor");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Check Scalar Base Case
  const LinearSeries *LS_AddScalar =
      Ripple->getLinseriesFor(cast<Instruction>(&*AddScalar));
  ASSERT_NE(LS_AddScalar, nullptr);
  EXPECT_TRUE(LS_AddScalar->getBaseShape().isScalar()); // Scalar base
  EXPECT_TRUE(LS_AddScalar->hasZeroSlopes());
  // Shape should be [4][8][1] - well, [4][8] in lower dims
  EXPECT_EQ(LS_AddScalar->getShape()[0], 4u);
  EXPECT_EQ(LS_AddScalar->getShape()[1], 8u);
  EXPECT_EQ(LS_AddScalar->getShape()[2], 1u);
  EXPECT_EQ(LS_AddScalar->getBase(), AddScalar);

  // Check Tensor Base Case
  const LinearSeries *LS_AddTensor =
      Ripple->getLinseriesFor(cast<Instruction>(&*AddTensor));
  ASSERT_NE(LS_AddTensor, nullptr);
  EXPECT_FALSE(LS_AddTensor->getBaseShape().isScalar());
  // Base shape [4] (others 1)
  EXPECT_EQ(LS_AddTensor->getBaseShape()[0], 4u);
  EXPECT_EQ(LS_AddTensor->getBaseShape()[1], 1u);
  EXPECT_EQ(LS_AddTensor->getBaseShape()[2], 1u);
  // Base should be the Add instruction (BinOp) because base shapes match
  EXPECT_EQ(LS_AddTensor->getBase(), AddTensor);
  // Zero slopes shape [1][8][16]
  EXPECT_TRUE(LS_AddTensor->hasZeroSlopes());
  EXPECT_EQ(LS_AddTensor->getSlopeShape()[0], 1u);
  EXPECT_EQ(LS_AddTensor->getSlopeShape()[1], 8u);
  EXPECT_EQ(LS_AddTensor->getSlopeShape()[2], 16u);

  // Verify shape is [4][8][16]
  EXPECT_EQ(LS_AddTensor->getShape()[0], 4u);
  EXPECT_EQ(LS_AddTensor->getShape()[1], 8u);
  EXPECT_EQ(LS_AddTensor->getShape()[2], 16u);

  // Check TensorBase
  const LinearSeries *LS_TensorBase =
      Ripple->getLinseriesFor(cast<Instruction>(&*TensorBase));
  ASSERT_NE(LS_TensorBase, nullptr);
  EXPECT_FALSE(LS_TensorBase->getBaseShape().isScalar());
  EXPECT_TRUE(LS_TensorBase->getSlopeShape().isScalar());
  EXPECT_EQ(LS_TensorBase->getBaseShape()[0], 4u);
  EXPECT_EQ(LS_TensorBase->getBaseShape()[1], 1u);
  EXPECT_EQ(LS_TensorBase->getBaseShape()[2], 1u);
  EXPECT_EQ(LS_TensorBase->getBase(), TensorBase);
  EXPECT_TRUE(LS_TensorBase->hasZeroSlopes());

  // Check BroadcastTensor1
  const LinearSeries *LS_BroadcastTensor1 =
      Ripple->getLinseriesFor(cast<Instruction>(&*BroadcastTensor1));
  ASSERT_NE(LS_BroadcastTensor1, nullptr);
  // Base should be TensorBase [4]
  EXPECT_FALSE(LS_BroadcastTensor1->getBaseShape().isScalar());
  EXPECT_EQ(LS_BroadcastTensor1->getBaseShape()[0], 4u);
  EXPECT_EQ(LS_BroadcastTensor1->getBaseShape()[1], 1u);
  EXPECT_EQ(LS_BroadcastTensor1->getBaseShape()[2], 1u);
  // Slope should be TensorBase [1][8]
  EXPECT_FALSE(LS_BroadcastTensor1->getSlopeShape().isScalar());
  EXPECT_EQ(LS_BroadcastTensor1->getSlopeShape()[0], 1u);
  EXPECT_EQ(LS_BroadcastTensor1->getSlopeShape()[1], 8u);
  EXPECT_EQ(LS_BroadcastTensor1->getSlopeShape()[2], 1u);
  // Shape should be [4][8][1]
  EXPECT_EQ(LS_BroadcastTensor1->getShape()[0], 4u);
  EXPECT_EQ(LS_BroadcastTensor1->getShape()[1], 8u);
  EXPECT_EQ(LS_BroadcastTensor1->getShape()[2], 1u);
  EXPECT_TRUE(LS_BroadcastTensor1->hasZeroSlopes());
}

TEST_F(PropagateLinearSeriesTest, ComplexLinearSeries) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [10][8] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(10);
  Value *Dim1 = IRB.getInt64(8);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 = IRB.CreateCall(IndexFn, {SetShape, Idx1Val});

  // Test: Complex expression: (index0 * 3 + 5) + (index1 * 2)
  // This should create a 2D LinearSeries with:
  // - base = complex expression
  // - slope[0] = 3
  // - slope[1] = 2
  Value *Scalar3 = IRB.getInt64(3);
  Value *Scalar5 = IRB.getInt64(5);
  Value *Scalar2 = IRB.getInt64(2);

  AssertingVH<Value> MulIndex0 = IRB.CreateMul(Index0, Scalar3, "mul_index0");
  AssertingVH<Value> AddScalar =
      IRB.CreateAdd(MulIndex0, Scalar5, "add_scalar");
  AssertingVH<Value> MulIndex1 = IRB.CreateMul(Index1, Scalar2, "mul_index1");
  AssertingVH<Value> FinalAdd =
      IRB.CreateAdd(AddScalar, MulIndex1, "final_add");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Verify the final complex LinearSeries
  const LinearSeries *LS_Final =
      Ripple->getLinseriesFor(cast<Instruction>(&*FinalAdd));
  ASSERT_NE(LS_Final, nullptr);

  // Base should be the final Add instruction
  EXPECT_EQ(LS_Final->getBase(), FinalAdd);
  EXPECT_TRUE(LS_Final->getBaseShape().isScalar());

  // Slope[0] should be constant 3
  if (auto *Slope0Const = dyn_cast<ConstantInt>(LS_Final->getSlope(0))) {
    EXPECT_EQ(Slope0Const->getZExtValue(), 3u);
  } else {
    FAIL() << "Slope[0] is not a ConstantInt";
  }

  // Slope[1] should be constant 2
  if (auto *Slope1Const = dyn_cast<ConstantInt>(LS_Final->getSlope(1))) {
    EXPECT_EQ(Slope1Const->getZExtValue(), 2u);
  } else {
    FAIL() << "Slope[1] is not a ConstantInt";
  }
}

// Default case: when we cannot propagate, we expect to restart a new LS with
// tensor base and scalar zero slope.
TEST_F(PropagateLinearSeriesTest, LinearSeriesRestart) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [10][10] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(10);
  Value *Dim1 = IRB.getInt64(10);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 = IRB.CreateCall(IndexFn, {SetShape, Idx1Val});

  // Case 1: Mul - Index * Index (Non-linear)
  AssertingVH<Value> MulNonLinear =
      IRB.CreateMul(Index0, Index0, "mul_nonlinear");

  // Case 2: Mul - Index0 * Index1 (Bilinear)
  AssertingVH<Value> MulBilinear =
      IRB.CreateMul(Index0, Index1, "mul_bilinear");

  // Case 3: Add - (Index0 * Index0) + Index0 (Orthogonality violation)
  AssertingVH<Value> AddOrthog =
      IRB.CreateAdd(MulNonLinear, Index0, "add_orthog");

  // Case 4: Sub - (Index0 * Index0) - Index0 (Orthogonality violation)
  AssertingVH<Value> SubOrthog =
      IRB.CreateSub(MulNonLinear, Index0, "sub_orthog");

  // Case 5: Add - (Index0 * Index0) + (Index1 * Index1) (Different tensor base
  // shapes)
  AssertingVH<Value> MulNonLinear2 =
      IRB.CreateMul(Index1, Index1, "mul_nonlinear2");
  AssertingVH<Value> AddTensorBases =
      IRB.CreateAdd(MulNonLinear, MulNonLinear2, "add_tensor_bases");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  Err = Ripple->createLinearSeries();
  ASSERT_FALSE(Err);

  // Check Case 1
  const LinearSeries *LS1 =
      Ripple->getLinseriesFor(cast<Instruction>(&*MulNonLinear));
  ASSERT_NE(LS1, nullptr);
  EXPECT_TRUE(LS1->getBaseShape().isVector()); // Base is tensor
  EXPECT_EQ(LS1->getBase(), MulNonLinear);     // Base is the instruction itself
  EXPECT_TRUE(LS1->getSlopeShape().isScalar());

  // Check Case 2
  const LinearSeries *LS2 =
      Ripple->getLinseriesFor(cast<Instruction>(&*MulBilinear));
  ASSERT_NE(LS2, nullptr);
  EXPECT_TRUE(LS2->getBaseShape().isVector()); // Base is tensor
  EXPECT_EQ(LS2->getBase(), MulBilinear);
  EXPECT_TRUE(LS2->getSlopeShape().isScalar());

  // Check Case 3
  const LinearSeries *LS3 =
      Ripple->getLinseriesFor(cast<Instruction>(&*AddOrthog));
  ASSERT_NE(LS3, nullptr);
  EXPECT_TRUE(LS3->getBaseShape().isVector());
  EXPECT_EQ(LS3->getBase(), AddOrthog);
  EXPECT_TRUE(LS3->getSlopeShape().isScalar());

  // Check Case 4
  const LinearSeries *LS4 =
      Ripple->getLinseriesFor(cast<Instruction>(&*SubOrthog));
  ASSERT_NE(LS4, nullptr);
  EXPECT_TRUE(LS4->getBaseShape().isVector());
  EXPECT_EQ(LS4->getBase(), SubOrthog);
  EXPECT_TRUE(LS4->getSlopeShape().isScalar());
  EXPECT_TRUE(LS4->hasZeroSlopes());

  // Check Case 5
  const LinearSeries *LS5 =
      Ripple->getLinseriesFor(cast<Instruction>(&*AddTensorBases));
  ASSERT_NE(LS5, nullptr);
  EXPECT_TRUE(LS5->getBaseShape().isVector());
  EXPECT_EQ(LS5->getBase(), AddTensorBases);
  EXPECT_TRUE(LS5->getSlopeShape().isScalar());
}

} // namespace
} // namespace llvm
