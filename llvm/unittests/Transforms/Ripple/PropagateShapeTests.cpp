//===- PropagateShapeTests.cpp - -----------===//
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

// Custom diagnostic handler that suppresses all diagnostics
class SuppressingDiagnosticHandler : public DiagnosticHandler {
public:
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    // Return true to suppress printing
    return true;
  }
};

using PropagateShapeTest = RippleFunctionTest;

TEST_F(PropagateShapeTest, ScalarFunction) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  // Create various scalar operations without any Ripple intrinsics
  Value *V1 = IRB.getInt32(10);
  Value *V2 = IRB.getInt32(20);
  Value *V3 = IRB.getInt32(5);

  // Arithmetic operations
  AssertingVH<Value> Add = IRB.CreateAdd(V1, V2, "add");   // 10 + 20
  AssertingVH<Value> Sub = IRB.CreateSub(Add, V3, "sub");  // 30 - 5
  AssertingVH<Value> Mul = IRB.CreateMul(Sub, V1, "mul");  // 25 * 10
  AssertingVH<Value> Div = IRB.CreateSDiv(Mul, V2, "div"); // 250 / 20

  // Comparison operations
  AssertingVH<Value> Cmp = IRB.CreateICmpSGT(Div, V3, "cmp"); // 12 > 5

  // Bitwise operations
  AssertingVH<Value> And = IRB.CreateAnd(V1, V2, "and");
  AssertingVH<Value> Or = IRB.CreateOr(And, V3, "or");
  AssertingVH<Value> Xor = IRB.CreateXor(Or, V1, "xor");

  // Unary operations
  AssertingVH<Value> Neg = IRB.CreateNeg(Xor, "neg");
  AssertingVH<Value> Not = IRB.CreateNot(Neg, "not");

  IRB.CreateRetVoid();

  // Verify that the generated IR is valid
  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);
  EXPECT_FALSE(WaitingForSpec);

  // Verify all operations result in scalar shapes
  const TensorShape &AddShape = Ripple->getRippleShape(Add);
  EXPECT_TRUE(AddShape.isScalar());

  const TensorShape &SubShape = Ripple->getRippleShape(Sub);
  EXPECT_TRUE(SubShape.isScalar());

  const TensorShape &MulShape = Ripple->getRippleShape(Mul);
  EXPECT_TRUE(MulShape.isScalar());

  const TensorShape &DivShape = Ripple->getRippleShape(Div);
  EXPECT_TRUE(DivShape.isScalar());

  const TensorShape &CmpShape = Ripple->getRippleShape(Cmp);
  EXPECT_TRUE(CmpShape.isScalar());

  const TensorShape &AndShape = Ripple->getRippleShape(And);
  EXPECT_TRUE(AndShape.isScalar());

  const TensorShape &OrShape = Ripple->getRippleShape(Or);
  EXPECT_TRUE(OrShape.isScalar());

  const TensorShape &XorShape = Ripple->getRippleShape(Xor);
  EXPECT_TRUE(XorShape.isScalar());

  const TensorShape &NegShape = Ripple->getRippleShape(Neg);
  EXPECT_TRUE(NegShape.isScalar());

  const TensorShape &NotShape = Ripple->getRippleShape(Not);
  EXPECT_TRUE(NotShape.isScalar());
}

TEST_F(PropagateShapeTest, RippleBlockIndexCreation) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  // Declare intrinsics
  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Create a [4][8] shape
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(4);
  Value *Dim1 = IRB.getInt64(8);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  // Create [4] shape
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 = IRB.CreateCall(IndexFn, {SetShape, Idx0Val});

  // Create [1][8] shape
  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 = IRB.CreateCall(IndexFn, {SetShape, Idx1Val});

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify shapes
  // Index0 should be [4]
  const TensorShape &Shape0 =
      Ripple->getRippleShape(static_cast<Value *>(Index0));
  EXPECT_EQ(Shape0.rank(), 2u);
  EXPECT_EQ(Shape0[0], 4u);
  EXPECT_EQ(Shape0[1], 1u);

  // Index1 should be [1][8]
  const TensorShape &Shape1 =
      Ripple->getRippleShape(static_cast<Value *>(Index1));
  EXPECT_EQ(Shape1.rank(), 2u);
  EXPECT_EQ(Shape1[0], 1u);
  EXPECT_EQ(Shape1[1], 8u);
}

TEST_F(PropagateShapeTest, UnaryOperatorPropagation) {
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

  // Create a tensor with shape [6][3]
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(6);
  Value *Dim1 = IRB.getInt64(3);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val}); // [6]

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape, Idx1Val}); // [1][3]

  // Create a 2D tensor [6][3] by broadcasting [6] + [1][3] -> [6][3]
  AssertingVH<Value> Tensor2D_6x3 =
      IRB.CreateAdd(Index0, Index1, "tensor_2d_6x3");

  // Test 1: Apply unary operator (negation) to 1D tensor: -[6] -> [6]
  AssertingVH<Value> Neg1D = IRB.CreateNeg(Index0, "neg_1d");

  // Test 2: Apply unary operator (negation) to 2D tensor: -[6][3] -> [6][3]
  AssertingVH<Value> Neg2D = IRB.CreateNeg(Tensor2D_6x3, "neg_2d");

  // Test 3: Apply unary operator (bitwise NOT) to 1D tensor: ~[6] -> [6]
  AssertingVH<Value> Not1D = IRB.CreateNot(Index0, "not_1d");

  // Test 4: Apply unary operator (bitwise NOT) to 2D tensor: ~[6][3] -> [6][3]
  AssertingVH<Value> Not2D = IRB.CreateNot(Tensor2D_6x3, "not_2d");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify original tensor shapes
  const TensorShape &Index0Shape =
      Ripple->getRippleShape(static_cast<Value *>(Index0));
  EXPECT_EQ(Index0Shape.rank(), 2u);
  EXPECT_EQ(Index0Shape[0], 6u);
  EXPECT_EQ(Index0Shape[1], 1u);

  const TensorShape &Index1Shape =
      Ripple->getRippleShape(static_cast<Value *>(Index1));
  EXPECT_EQ(Index1Shape.rank(), 2u);
  EXPECT_EQ(Index1Shape[0], 1u);
  EXPECT_EQ(Index1Shape[1], 3u);

  // Verify Test 1: 1D negation preserves shape [6]
  const TensorShape &Neg1DShape = Ripple->getRippleShape(Neg1D);
  EXPECT_EQ(Neg1DShape.rank(), 2u);
  EXPECT_EQ(Neg1DShape[0], 6u);
  EXPECT_EQ(Neg1DShape[1], 1u);

  // Verify Test 2: 2D negation preserves shape [6][3]
  const TensorShape &Neg2DShape = Ripple->getRippleShape(Neg2D);
  EXPECT_EQ(Neg2DShape.rank(), 2u);
  EXPECT_EQ(Neg2DShape[0], 6u);
  EXPECT_EQ(Neg2DShape[1], 3u);

  // Verify Test 3: 1D bitwise NOT preserves shape [6]
  const TensorShape &Not1DShape = Ripple->getRippleShape(Not1D);
  EXPECT_EQ(Not1DShape.rank(), 2u);
  EXPECT_EQ(Not1DShape[0], 6u);
  EXPECT_EQ(Not1DShape[1], 1u);

  // Verify Test 4: 2D bitwise NOT preserves shape [6][3]
  const TensorShape &Not2DShape = Ripple->getRippleShape(Not2D);
  EXPECT_EQ(Not2DShape.rank(), 2u);
  EXPECT_EQ(Not2DShape[0], 6u);
  EXPECT_EQ(Not2DShape[1], 3u);
}

TEST_F(PropagateShapeTest, TensorBroadcast2D) {
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

  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(4);
  Value *Dim1 = IRB.getInt64(8);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val}); // [4]

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape, Idx1Val}); // [1][8]

  // Test 1: Tensor broadcast - [4] + [1][8] -> [4][8]
  AssertingVH<Value> TensorBroadcast =
      IRB.CreateAdd(Index0, Index1, "tensor_broadcast");

  // Test 2: Scalar + Tensor -> Tensor
  Value *Scalar = IRB.getInt64(42);
  AssertingVH<Value> ScalarPlusTensor =
      IRB.CreateAdd(Scalar, Index0, "scalar_plus_tensor");

  // Test 3: Tensor + Scalar -> Tensor
  AssertingVH<Value> TensorPlusScalar =
      IRB.CreateAdd(Index1, Scalar, "tensor_plus_scalar");

  // Test 4: Same shape tensors - [4] + [4] -> [4]
  Value *Idx0Val2 = IRB.getInt64(0);
  AssertingVH<CallInst> Index0_2 =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val2}); // [4]
  AssertingVH<Value> SameShapeTensors =
      IRB.CreateAdd(Index0, Index0_2, "same_shape_tensors");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify tensor broadcast: [4] + [1][8] -> [4][8]
  const TensorShape &BroadcastShape = Ripple->getRippleShape(TensorBroadcast);
  EXPECT_EQ(BroadcastShape.rank(), 2u);
  EXPECT_EQ(BroadcastShape[0], 4u);
  EXPECT_EQ(BroadcastShape[1], 8u);

  // Verify scalar + tensor: scalar + [4] -> [4]
  const TensorShape &ScalarPlusTensorShape =
      Ripple->getRippleShape(ScalarPlusTensor);
  EXPECT_EQ(ScalarPlusTensorShape.rank(), 2u);
  EXPECT_EQ(ScalarPlusTensorShape[0], 4u);
  EXPECT_EQ(ScalarPlusTensorShape[1], 1u);

  // Verify tensor + scalar: [1][8] + scalar -> [1][8]
  const TensorShape &TensorPlusScalarShape =
      Ripple->getRippleShape(TensorPlusScalar);
  EXPECT_EQ(TensorPlusScalarShape.rank(), 2u);
  EXPECT_EQ(TensorPlusScalarShape[0], 1u);
  EXPECT_EQ(TensorPlusScalarShape[1], 8u);

  // Verify same shape tensors: [4] + [4] -> [4]
  const TensorShape &SameShapeShape = Ripple->getRippleShape(SameShapeTensors);
  EXPECT_EQ(SameShapeShape.rank(), 2u);
  EXPECT_EQ(SameShapeShape[0], 4u);
  EXPECT_EQ(SameShapeShape[1], 1u);
}

TEST_F(PropagateShapeTest, CallInstBroadcast) {
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

  // Create tensors with different shapes
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(3);
  Value *Dim1 = IRB.getInt64(5);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val}); // [3]

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape, Idx1Val}); // [1][5]

  // Create an external function with three arguments
  Type *Int64Ty = IRB.getInt64Ty();
  FunctionType *ExternalFTy =
      FunctionType::get(Int64Ty, {Int64Ty, Int64Ty, Int64Ty}, false);
  Function *ExternalFn =
      Function::Create(ExternalFTy, GlobalValue::LinkageTypes::ExternalLinkage,
                       "external_func", M);

  Value *Scalar1 = IRB.getInt64(10);
  Value *Scalar2 = IRB.getInt64(20);

  // Test 1: (Tensor, scalar, scalar) -> Tensor
  // [3] with scalars -> [3]
  AssertingVH<Value> Call1 =
      IRB.CreateCall(ExternalFn, {Index0, Scalar1, Scalar2}, "call1");

  // Test 2: (scalar, scalar, scalar) -> scalar
  AssertingVH<Value> Call2 =
      IRB.CreateCall(ExternalFn, {Scalar1, Scalar2, Scalar1}, "call2");

  // Test 3: (scalar, scalar, tensor) -> Tensor
  // scalars with [1][5] -> [1][5]
  AssertingVH<Value> Call3 =
      IRB.CreateCall(ExternalFn, {Scalar1, Scalar2, Index1}, "call3");

  // Test 4a: (tensor, scalar, tensor) with same tensor -> same shape
  // [3] with scalar and [3] -> [3]
  Value *Idx0Val2 = IRB.getInt64(0);
  AssertingVH<CallInst> Index0_2 =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val2}); // [3]
  AssertingVH<Value> Call4a =
      IRB.CreateCall(ExternalFn, {Index0, Scalar1, Index0_2}, "call4a");

  // Test 4b: (tensor, scalar, tensor) with broadcast case
  // [3] with scalar and [1][5] -> [3][5]
  AssertingVH<Value> Call4b =
      IRB.CreateCall(ExternalFn, {Index0, Scalar1, Index1}, "call4b");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify Test 1: (Tensor, scalar, scalar) -> [3]
  const TensorShape &Call1Shape = Ripple->getRippleShape(Call1);
  EXPECT_EQ(Call1Shape.rank(), 2u);
  EXPECT_EQ(Call1Shape[0], 3u);
  EXPECT_EQ(Call1Shape[1], 1u);

  // Verify Test 2: (scalar, scalar, scalar) -> scalar
  const TensorShape &Call2Shape = Ripple->getRippleShape(Call2);
  EXPECT_TRUE(Call2Shape.isScalar());

  // Verify Test 3: (scalar, scalar, tensor) -> [1][5]
  const TensorShape &Call3Shape = Ripple->getRippleShape(Call3);
  EXPECT_EQ(Call3Shape.rank(), 2u);
  EXPECT_EQ(Call3Shape[0], 1u);
  EXPECT_EQ(Call3Shape[1], 5u);

  // Verify Test 4a: (tensor, scalar, tensor) same shape -> [3]
  const TensorShape &Call4aShape = Ripple->getRippleShape(Call4a);
  EXPECT_EQ(Call4aShape.rank(), 2u);
  EXPECT_EQ(Call4aShape[0], 3u);
  EXPECT_EQ(Call4aShape[1], 1u);

  // Verify Test 4b: (tensor, scalar, tensor) broadcast -> [3][5]
  const TensorShape &Call4bShape = Ripple->getRippleShape(Call4b);
  EXPECT_EQ(Call4bShape.rank(), 2u);
  EXPECT_EQ(Call4bShape[0], 3u);
  EXPECT_EQ(Call4bShape[1], 5u);
}

TEST_F(PropagateShapeTest, PHINodeBroadcast) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);

  // Create basic blocks: Entry -> A | B, A -> C | D, B -> D, C -> D
  BasicBlock *EntryBB = BasicBlock::Create(C, "Entry", F);
  BasicBlock *ABB = BasicBlock::Create(C, "A", F);
  BasicBlock *BBB = BasicBlock::Create(C, "B", F);
  BasicBlock *CBB = BasicBlock::Create(C, "C", F);
  BasicBlock *DBB = BasicBlock::Create(C, "D", F);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Entry Block: Set up tensors and branch to A or B
  IRB.SetInsertPoint(EntryBB);
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(4);
  Value *Dim1 = IRB.getInt64(6);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val}); // [4]

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape, Idx1Val}); // [1][6]

  Value *Scalar = IRB.getInt64(42);
  Value *Condition = IRB.getInt1(true);
  IRB.CreateCondBr(Condition, ABB, BBB);

  // Block A: Create tensor operations and branch to C or D
  IRB.SetInsertPoint(ABB);
  // Test case 1: Tensor + Scalar -> [4]
  AssertingVH<Value> ATensorScalar =
      IRB.CreateAdd(Index0, Scalar, "a_tensor_scalar");
  // Test case 2: Scalar + Tensor -> [1][6]
  AssertingVH<Value> AScalarTensor =
      IRB.CreateAdd(Scalar, Index1, "a_scalar_tensor");
  Value *ACondition = IRB.getInt1(false);
  IRB.CreateCondBr(ACondition, CBB, DBB);

  // Block B: Create different tensor operations and branch to C or D
  IRB.SetInsertPoint(BBB);
  // Test case 3: Tensor broadcast [4] + [1][6] -> [4][6]
  AssertingVH<Value> BTensorBroadcast =
      IRB.CreateAdd(Index0, Index1, "b_tensor_broadcast");
  // Test case 4: Same shape tensors [4][1] + [4] -> [4]
  Value *Idx0Val2 = IRB.getInt64(0);
  AssertingVH<CallInst> Index0_2 =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val2}); // [4]
  AssertingVH<Value> BSameShape =
      IRB.CreateAdd(Index0, Index0_2, "b_same_shape");
  Value *BCondition = IRB.getInt1(true);
  IRB.CreateCondBr(BCondition, CBB, DBB);

  // Block C: PHI nodes to merge values from A and B, then create operations and
  // branch to D
  IRB.SetInsertPoint(CBB);

  // PHI 1 in C: (Tensor + Scalar) from A vs (Tensor broadcast) from B -> [4]
  // vs [4][6] -> [4][6]
  AssertingVH<PHINode> PhiC1 = IRB.CreatePHI(IRB.getInt64Ty(), 2, "phi_c1");
  PhiC1->addIncoming(ATensorScalar, ABB);    // [4] from A
  PhiC1->addIncoming(BTensorBroadcast, BBB); // [4][6] from B

  // PHI 2 in C: (Scalar + Tensor) from A vs (Same shape) from B -> [1][6] vs
  // [4] -> [4][6]
  AssertingVH<PHINode> PhiC2 = IRB.CreatePHI(IRB.getInt64Ty(), 2, "phi_c2");
  PhiC2->addIncoming(AScalarTensor, ABB); // [1][6] from A
  PhiC2->addIncoming(BSameShape, BBB);    // [4] from B

  // Create a [4] shape value in block C for Phi4 testing
  AssertingVH<Value> CTensor41 =
      IRB.CreateAdd(Index0, Scalar, "c_tensor_41"); // [4] + scalar -> [4]

  // Test case 5: Scalar + Scalar -> Scalar
  AssertingVH<Value> CScalarScalar =
      IRB.CreateAdd(Scalar, Scalar, "c_scalar_scalar");
  IRB.CreateBr(DBB);

  // Block D: PHI nodes to merge values from A, B, and C
  IRB.SetInsertPoint(DBB);

  // PHI 1 in D: Values from A, B, and C
  AssertingVH<PHINode> Phi1 = IRB.CreatePHI(IRB.getInt64Ty(), 3, "phi1");
  Phi1->addIncoming(ATensorScalar, ABB);    // [4] from A when A->D
  Phi1->addIncoming(BTensorBroadcast, BBB); // [4][6] from B when B->D
  Phi1->addIncoming(PhiC1, CBB); // [4][6] from C (result of PHI in C)

  // PHI 2 in D: Different values from A, B, and C
  AssertingVH<PHINode> Phi2 = IRB.CreatePHI(IRB.getInt64Ty(), 3, "phi2");
  Phi2->addIncoming(AScalarTensor, ABB); // [1][6] from A when A->D
  Phi2->addIncoming(BSameShape, BBB);    // [4] from B when B->D
  Phi2->addIncoming(PhiC2, CBB);         // [4][6] from C (result of PHI in C)

  // PHI 3 in D: All scalars from A, B, and C
  AssertingVH<PHINode> Phi3 = IRB.CreatePHI(IRB.getInt64Ty(), 3, "phi3");
  Phi3->addIncoming(Scalar, ABB);        // scalar from A when A->D
  Phi3->addIncoming(Scalar, BBB);        // scalar from B when B->D
  Phi3->addIncoming(CScalarScalar, CBB); // scalar from C

  // PHI 4 in D: Test CTensor41 ([4]) with tensor [4] -> should remain
  // [4]
  AssertingVH<PHINode> Phi4 = IRB.CreatePHI(IRB.getInt64Ty(), 3, "phi4");
  Phi4->addIncoming(ATensorScalar, ABB); // [4] from A when A->D
  Phi4->addIncoming(BSameShape, BBB);    // [4] from B when B->D
  Phi4->addIncoming(CTensor41, CBB);     // [4] from C (CTensor41 result)

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify PHI C1 in block C: Should broadcast [4] and [4][6] -> [4][6]
  const TensorShape &PhiC1Shape =
      Ripple->getRippleShape(static_cast<Value *>(PhiC1));
  EXPECT_EQ(PhiC1Shape.rank(), 2u);
  EXPECT_EQ(PhiC1Shape[0], 4u);
  EXPECT_EQ(PhiC1Shape[1], 6u);

  // Verify PHI C2 in block C: Should broadcast [1][6] and [4] -> [4][6]
  const TensorShape &PhiC2Shape =
      Ripple->getRippleShape(static_cast<Value *>(PhiC2));
  EXPECT_EQ(PhiC2Shape.rank(), 2u);
  EXPECT_EQ(PhiC2Shape[0], 4u);
  EXPECT_EQ(PhiC2Shape[1], 6u);

  // Verify PHI 1 in block D: Should broadcast [4] and [4][6] -> [4][6]
  const TensorShape &Phi1Shape =
      Ripple->getRippleShape(static_cast<Value *>(Phi1));
  EXPECT_EQ(Phi1Shape.rank(), 2u);
  EXPECT_EQ(Phi1Shape[0], 4u);
  EXPECT_EQ(Phi1Shape[1], 6u);

  // Verify PHI 2 in block D: Should broadcast [1][6] and [4] -> [4][6]
  const TensorShape &Phi2Shape =
      Ripple->getRippleShape(static_cast<Value *>(Phi2));
  EXPECT_EQ(Phi2Shape.rank(), 2u);
  EXPECT_EQ(Phi2Shape[0], 4u);
  EXPECT_EQ(Phi2Shape[1], 6u);

  // Verify PHI 3 in block D: All scalars -> scalar
  const TensorShape &Phi3Shape =
      Ripple->getRippleShape(static_cast<Value *>(Phi3));
  EXPECT_TRUE(Phi3Shape.isScalar());

  // Verify PHI 4 in block D: All [4] inputs -> [4]
  const TensorShape &Phi4Shape =
      Ripple->getRippleShape(static_cast<Value *>(Phi4));
  EXPECT_EQ(Phi4Shape.rank(), 2u);
  EXPECT_EQ(Phi4Shape[0], 4u);
  EXPECT_EQ(Phi4Shape[1], 1u);
}

TEST_F(PropagateShapeTest, LoopPhiPropagation) {
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

  // Entry Block: Set up initial values and jump to loop header
  IRB.SetInsertPoint(EntryBB);
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(4000);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  // Get ripple_id for accumulation
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> RippleId =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val}); // [4000]

  // Initial accumulator value (scalar)
  Value *InitAccum = IRB.getInt64(0);

  // Initial loop counter
  Value *InitCounter = IRB.getInt64(0);
  Value *LoopBound = IRB.getInt64(42);

  IRB.CreateBr(LoopHeaderBB);

  // Loop Header: PHI nodes for accumulator and counter, loop condition check
  IRB.SetInsertPoint(LoopHeaderBB);

  // PHI for accumulator: starts with InitAccum, updated in loop latch
  AssertingVH<PHINode> AccumPhi =
      IRB.CreatePHI(IRB.getInt64Ty(), 2, "accum_phi");
  AccumPhi->addIncoming(InitAccum, EntryBB);

  // PHI for loop counter: starts with InitCounter, incremented in loop latch
  AssertingVH<PHINode> CounterPhi =
      IRB.CreatePHI(IRB.getInt64Ty(), 2, "counter_phi");
  CounterPhi->addIncoming(InitCounter, EntryBB);

  // Loop condition: counter < 42
  AssertingVH<Value> LoopCond =
      IRB.CreateICmpULT(CounterPhi, LoopBound, "loop_cond");
  IRB.CreateCondBr(LoopCond, LoopBodyBB, ExitBB);

  // Loop Body: Accumulate ripple_id value
  IRB.SetInsertPoint(LoopBodyBB);

  // Add ripple_id to accumulator: accum = accum + ripple_id
  // This tests tensor + scalar -> tensor shape propagation in a loop context
  AssertingVH<Value> NewAccum = IRB.CreateAdd(AccumPhi, RippleId, "new_accum");

  // Test: Create another tensor operation in the loop body
  AssertingVH<Value> TensorOp =
      IRB.CreateMul(RippleId, IRB.getInt64(2),
                    "tensor_op"); // [4000] * scalar -> [4000]

  IRB.CreateBr(LoopLatchBB);

  // Loop Latch: Increment counter and branch back to header
  IRB.SetInsertPoint(LoopLatchBB);

  // Increment counter
  AssertingVH<Value> NextCounter =
      IRB.CreateAdd(CounterPhi, One, "next_counter");

  // Update PHI incoming values
  AccumPhi->addIncoming(NewAccum, LoopLatchBB);
  CounterPhi->addIncoming(NextCounter, LoopLatchBB);

  IRB.CreateBr(LoopHeaderBB);

  // Exit Block: Return void
  IRB.SetInsertPoint(ExitBB);
  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify shapes in the loop context

  // Verify RippleId shape: should be [4000]
  const TensorShape &RippleIdShape =
      Ripple->getRippleShape(static_cast<const Value *>(RippleId));
  EXPECT_EQ(RippleIdShape.rank(), 1u);
  EXPECT_EQ(RippleIdShape[0], 4000u);

  // Verify AccumPhi shape: scalar + tensor -> tensor, so should be [4000]
  const TensorShape &AccumPhiShape =
      Ripple->getRippleShape(static_cast<const Value *>(AccumPhi));
  EXPECT_EQ(AccumPhiShape.rank(), 1u);
  EXPECT_EQ(AccumPhiShape[0], 4000u);

  // Verify NewAccum shape: tensor + tensor -> tensor, should be [4000]
  const TensorShape &NewAccumShape = Ripple->getRippleShape(NewAccum);
  EXPECT_EQ(NewAccumShape.rank(), 1u);
  EXPECT_EQ(NewAccumShape[0], 4000u);

  // Verify TensorOp shape: tensor * scalar -> tensor, should be [4000]
  const TensorShape &TensorOpShape = Ripple->getRippleShape(TensorOp);
  EXPECT_EQ(TensorOpShape.rank(), 1u);
  EXPECT_EQ(TensorOpShape[0], 4000u);

  // Verify CounterPhi shape: should be scalar
  const TensorShape &CounterPhiShape =
      Ripple->getRippleShape(static_cast<const Value *>(CounterPhi));
  EXPECT_TRUE(CounterPhiShape.isScalar());

  // Verify NextCounter shape: should be scalar
  const TensorShape &NextCounterShape = Ripple->getRippleShape(NextCounter);
  EXPECT_TRUE(NextCounterShape.isScalar());
}

TEST_F(PropagateShapeTest, BroadcastFailure) {
  // Set up diagnostic handler to suppress diagnostics output to avoid polluting
  // the test stderr for the expected error
  auto OriginalHandler = C.getDiagnosticHandler();
  C.setDiagnosticHandler(std::make_unique<SuppressingDiagnosticHandler>());

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

  // Create tensors with truly incompatible shapes for broadcasting
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(3);
  Value *Dim1 = IRB.getInt64(4);
  Value *Dim2 = IRB.getInt64(5);
  Value *One = IRB.getInt64(1);

  // First tensor setup: [3][4]
  AssertingVH<CallInst> SetShape1 = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape1, Idx0Val}); // [3]

  // Second tensor setup: [5][2] - incompatible with [3][4]
  Value *Dim3 = IRB.getInt64(2);
  AssertingVH<CallInst> SetShape2 = IRB.CreateCall(
      SetShapeFn, {PE, Dim2, Dim3, One, One, One, One, One, One, One, One});

  Value *Idx1Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape2, Idx1Val}); // [5]

  // Test: Try to add tensors with truly incompatible shapes
  // [3] + [5] should fail because 3 != 5 in the first dimension
  // Neither dimension is 1, so they cannot be broadcasted together
  IRB.CreateAdd(Index0, Index1, "incompatible_add");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);

  // Expect shape propagation to fail due to incompatible broadcast shapes
  ASSERT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));

  // Restore original diagnostic handler
  if (OriginalHandler) {
    C.setDiagnosticHandler(std::move(OriginalHandler));
  }
}

TEST_F(PropagateShapeTest, CallInstBroadcastFailure) {
  // Set up diagnostic handler to suppress diagnostics output to avoid polluting
  // the test stderr for the expected error
  auto OriginalHandler = C.getDiagnosticHandler();
  C.setDiagnosticHandler(std::make_unique<SuppressingDiagnosticHandler>());

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

  // Create tensors with incompatible shapes for function call broadcasting
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(3);
  Value *Dim1 = IRB.getInt64(7);
  Value *One = IRB.getInt64(1);

  // First tensor setup: [3]
  AssertingVH<CallInst> SetShape1 = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape1, Idx0Val}); // [3]

  // Second tensor setup: [7] - incompatible with [3] for broadcasting
  AssertingVH<CallInst> SetShape2 = IRB.CreateCall(
      SetShapeFn, {PE, Dim1, One, One, One, One, One, One, One, One, One});
  Value *Idx1Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape2, Idx1Val}); // [7]

  // Create an external function with two tensor arguments
  Type *Int64Ty = IRB.getInt64Ty();
  FunctionType *ExternalFTy =
      FunctionType::get(Int64Ty, {Int64Ty, Int64Ty}, false);
  Function *ExternalFn =
      Function::Create(ExternalFTy, GlobalValue::LinkageTypes::ExternalLinkage,
                       "external_func", M);

  // Test: Try to call function with incompatible tensor shapes
  // [3] and [7] should fail because 3 != 7 in the first dimension
  // Neither dimension is 1, so they cannot be broadcasted together
  IRB.CreateCall(ExternalFn, {Index0, Index1}, "incompatible_call");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);

  // Expect shape propagation to fail due to incompatible broadcast shapes in
  // call
  ASSERT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));

  // Restore original diagnostic handler
  if (OriginalHandler) {
    C.setDiagnosticHandler(std::move(OriginalHandler));
  }
}

TEST_F(PropagateShapeTest, PHINodeBroadcastFailure) {
  // Set up diagnostic handler to suppress diagnostics output to avoid polluting
  // the test stderr for the expected error
  auto OriginalHandler = C.getDiagnosticHandler();
  C.setDiagnosticHandler(std::make_unique<SuppressingDiagnosticHandler>());

  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);

  // Create basic blocks: Entry -> A | B -> Merge
  BasicBlock *EntryBB = BasicBlock::Create(C, "Entry", F);
  BasicBlock *ABB = BasicBlock::Create(C, "A", F);
  BasicBlock *BBB = BasicBlock::Create(C, "B", F);
  BasicBlock *MergeBB = BasicBlock::Create(C, "Merge", F);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Entry Block: Set up tensors and branch to A or B
  IRB.SetInsertPoint(EntryBB);
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(4);
  Value *Dim1 = IRB.getInt64(9);
  Value *One = IRB.getInt64(1);

  // First tensor setup: [4]
  AssertingVH<CallInst> SetShape1 = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape1, Idx0Val}); // [4]

  // Second tensor setup: [9] - incompatible with [4]
  AssertingVH<CallInst> SetShape2 = IRB.CreateCall(
      SetShapeFn, {PE, Dim1, One, One, One, One, One, One, One, One, One});
  Value *Idx1Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape2, Idx1Val}); // [9]

  Value *Condition = IRB.getInt1(true);
  IRB.CreateCondBr(Condition, ABB, BBB);

  // Block A: Use first tensor
  IRB.SetInsertPoint(ABB);
  AssertingVH<Value> ATensor =
      IRB.CreateAdd(Index0, IRB.getInt64(1), "a_tensor"); // [4]
  IRB.CreateBr(MergeBB);

  // Block B: Use second tensor
  IRB.SetInsertPoint(BBB);
  AssertingVH<Value> BTensor =
      IRB.CreateAdd(Index1, IRB.getInt64(2), "b_tensor"); // [9]
  IRB.CreateBr(MergeBB);

  // Merge Block: PHI node with incompatible tensor shapes
  IRB.SetInsertPoint(MergeBB);
  // Test: Try to create PHI with incompatible tensor shapes
  // [4] from A and [9] from B should fail because 4 != 9
  // Neither dimension is 1, so they cannot be broadcasted together
  AssertingVH<PHINode> IncompatiblePhi =
      IRB.CreatePHI(IRB.getInt64Ty(), 2, "incompatible_phi");
  IncompatiblePhi->addIncoming(ATensor, ABB); // [4] from A
  IncompatiblePhi->addIncoming(BTensor, BBB); // [9] from B

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);

  // Expect shape propagation to fail due to incompatible broadcast shapes in
  // PHI
  ASSERT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));

  // Restore original diagnostic handler
  if (OriginalHandler) {
    C.setDiagnosticHandler(std::move(OriginalHandler));
  }
}

TEST_F(PropagateShapeTest, LoopPHIBroadcastFailure) {
  // Set up diagnostic handler to suppress diagnostics output to avoid polluting
  // the test stderr for the expected error
  auto OriginalHandler = C.getDiagnosticHandler();
  C.setDiagnosticHandler(std::make_unique<SuppressingDiagnosticHandler>());

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

  // Entry Block: Set up initial values and jump to loop header
  IRB.SetInsertPoint(EntryBB);
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(6);
  Value *Dim1 = IRB.getInt64(11);
  Value *One = IRB.getInt64(1);

  // First tensor setup: [6]
  AssertingVH<CallInst> SetShape1 = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape1, Idx0Val}); // [6]

  // Second tensor setup: [11] - incompatible with [6]
  AssertingVH<CallInst> SetShape2 = IRB.CreateCall(
      SetShapeFn, {PE, Dim1, One, One, One, One, One, One, One, One, One});
  Value *Idx1Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape2, Idx1Val}); // [11]

  // Initial accumulator value: use first tensor [6]
  AssertingVH<Value> InitAccum =
      IRB.CreateAdd(Index0, IRB.getInt64(0), "init_accum");

  // Initial loop counter
  Value *InitCounter = IRB.getInt64(0);
  Value *LoopBound = IRB.getInt64(2);

  IRB.CreateBr(LoopHeaderBB);

  // Loop Header: PHI nodes for accumulator and counter, loop condition check
  IRB.SetInsertPoint(LoopHeaderBB);

  // PHI for accumulator: starts with InitAccum [6], but will be updated with
  // incompatible shape
  AssertingVH<PHINode> AccumPhi =
      IRB.CreatePHI(IRB.getInt64Ty(), 2, "accum_phi");
  AccumPhi->addIncoming(InitAccum, EntryBB);

  // PHI for loop counter: starts with InitCounter, incremented in loop latch
  AssertingVH<PHINode> CounterPhi =
      IRB.CreatePHI(IRB.getInt64Ty(), 2, "counter_phi");
  CounterPhi->addIncoming(InitCounter, EntryBB);

  // Loop condition: counter < 2
  AssertingVH<Value> LoopCond =
      IRB.CreateICmpULT(CounterPhi, LoopBound, "loop_cond");
  IRB.CreateCondBr(LoopCond, LoopBodyBB, ExitBB);

  // Loop Body: Create incompatible tensor operation
  IRB.SetInsertPoint(LoopBodyBB);

  // Test: Try to create a new accumulator with incompatible shape
  // This will create a PHI node that tries to merge [6] (from entry) with [11]
  // (from loop) [6] and [11] should fail because 6 != 11 in the first dimension
  AssertingVH<Value> NewAccum =
      IRB.CreateAdd(Index1, IRB.getInt64(1), "new_accum"); // [11]

  IRB.CreateBr(LoopLatchBB);

  // Loop Latch: Increment counter and branch back to header
  IRB.SetInsertPoint(LoopLatchBB);

  // Increment counter
  AssertingVH<Value> NextCounter =
      IRB.CreateAdd(CounterPhi, One, "next_counter");

  // Update PHI incoming values - this will create the incompatible broadcast
  AccumPhi->addIncoming(NewAccum, LoopLatchBB); // [11] from loop latch
  CounterPhi->addIncoming(NextCounter, LoopLatchBB);

  IRB.CreateBr(LoopHeaderBB);

  // Exit Block: Return void
  IRB.SetInsertPoint(ExitBB);
  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);

  // Expect shape propagation to fail due to incompatible broadcast shapes in
  // loop PHI
  ASSERT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));

  // Restore original diagnostic handler
  if (OriginalHandler) {
    C.setDiagnosticHandler(std::move(OriginalHandler));
  }
}

TEST_F(PropagateShapeTest, GEPBroadcast) {
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

  // Create tensors with different shapes for testing
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(4);
  Value *Dim1 = IRB.getInt64(8);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, Dim1, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val}); // [4]

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape, Idx1Val}); // [1][8]

  // Create array types and allocate arrays for GEP testing
  Type *Int64Ty = IRB.getInt64Ty();
  Type *ArrayTy = ArrayType::get(Int64Ty, 100); // Array of 100 i64 elements
  Type *PtrTy = PointerType::get(C, 0);

  // Create some base addresses (scalars and tensors)
  AssertingVH<Value> ScalarAddr =
      IRB.CreateAlloca(ArrayTy, nullptr, "scalar_addr");
  AssertingVH<Value> TensorAddr4 =
      IRB.CreateAdd(Index0, IRB.getInt64(20),
                    "tensor_addr4"); // [4] - simulated tensor address
  AssertingVH<Value> TensorAddr18 =
      IRB.CreateAdd(Index1, IRB.getInt64(10),
                    "tensor_addr18"); // [1][8] - simulated tensor address

  // Create scalar indices
  Value *ScalarIdx = IRB.getInt64(5);

  // Test 1: Scalar address + Scalar index -> Scalar result
  AssertingVH<Value> GEP1 = IRB.CreateGEP(
      ArrayTy, ScalarAddr, {IRB.getInt64(0), ScalarIdx}, "gep_scalar_scalar");

  // Test 2: Tensor address + Scalar index -> Tensor result (same shape as
  // address)
  AssertingVH<Value> TensorAddrPtr4 =
      IRB.CreateIntToPtr(TensorAddr4, PtrTy, "tensor_addr_ptr4");
  AssertingVH<Value> GEP2 =
      IRB.CreateGEP(Int64Ty, TensorAddrPtr4, ScalarIdx, "gep_tensor_scalar");

  // Test 3: Scalar address + Tensor index -> Tensor result (same shape as
  // index)
  AssertingVH<Value> GEP3 = IRB.CreateGEP(
      ArrayTy, ScalarAddr, {IRB.getInt64(0), static_cast<Value *>(Index0)},
      "gep_scalar_tensor");

  // Test 4: Tensor address + Tensor index (same shape) -> Same tensor shape
  AssertingVH<Value> TensorAddrPtr18 =
      IRB.CreateIntToPtr(TensorAddr18, PtrTy, "tensor_addr_ptr18");
  AssertingVH<Value> GEP4 =
      IRB.CreateGEP(Int64Ty, TensorAddrPtr18, static_cast<Value *>(Index1),
                    "gep_tensor_tensor_same");

  // Test 5: Tensor address + Tensor index (broadcast case) -> Broadcast result
  // [4] address + [1][8] index -> [4][8]
  AssertingVH<Value> GEP5 =
      IRB.CreateGEP(Int64Ty, TensorAddrPtr4, static_cast<Value *>(Index1),
                    "gep_tensor_tensor_broadcast");

  // Test 6: Tensor address + Tensor index (broadcast case reverse) -> Broadcast
  // result [1][8] address + [4] index -> [4][8]
  AssertingVH<Value> GEP6 =
      IRB.CreateGEP(Int64Ty, TensorAddrPtr18, static_cast<Value *>(Index0),
                    "gep_tensor_tensor_broadcast_reverse");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify Test 1: Scalar address + Scalar index -> Scalar
  const TensorShape &GEP1Shape = Ripple->getRippleShape(GEP1);
  EXPECT_TRUE(GEP1Shape.isScalar());

  // Verify Test 2: Tensor address [4] + Scalar index -> [4]
  const TensorShape &GEP2Shape = Ripple->getRippleShape(GEP2);
  EXPECT_EQ(GEP2Shape.rank(), 2u);
  EXPECT_EQ(GEP2Shape[0], 4u);
  EXPECT_EQ(GEP2Shape[1], 1u);

  // Verify Test 3: Scalar address + Tensor index [4] -> [4]
  const TensorShape &GEP3Shape = Ripple->getRippleShape(GEP3);
  EXPECT_EQ(GEP3Shape.rank(), 2u);
  EXPECT_EQ(GEP3Shape[0], 4u);
  EXPECT_EQ(GEP3Shape[1], 1u);

  // Verify Test 4: Tensor address [1][8] + Tensor index [1][8] -> [1][8]
  const TensorShape &GEP4Shape = Ripple->getRippleShape(GEP4);
  EXPECT_EQ(GEP4Shape.rank(), 2u);
  EXPECT_EQ(GEP4Shape[0], 1u);
  EXPECT_EQ(GEP4Shape[1], 8u);

  // Verify Test 5: Tensor address [4] + Tensor index [1][8] -> [4][8]
  // (broadcast)
  const TensorShape &GEP5Shape = Ripple->getRippleShape(GEP5);
  EXPECT_EQ(GEP5Shape.rank(), 2u);
  EXPECT_EQ(GEP5Shape[0], 4u);
  EXPECT_EQ(GEP5Shape[1], 8u);

  // Verify Test 6: Tensor address [1][8] + Tensor index [4] -> [4][8]
  // (broadcast)
  const TensorShape &GEP6Shape = Ripple->getRippleShape(GEP6);
  EXPECT_EQ(GEP6Shape.rank(), 2u);
  EXPECT_EQ(GEP6Shape[0], 4u);
  EXPECT_EQ(GEP6Shape[1], 8u);
}

TEST_F(PropagateShapeTest, GEPBroadcastFailure) {
  // Set up diagnostic handler to suppress diagnostics output to avoid polluting
  // the test stderr for the expected error
  auto OriginalHandler = C.getDiagnosticHandler();
  C.setDiagnosticHandler(std::make_unique<SuppressingDiagnosticHandler>());

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

  // Create tensors with incompatible shapes for GEP broadcasting
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(3);
  Value *Dim1 = IRB.getInt64(7);
  Value *One = IRB.getInt64(1);

  // First tensor setup: [3]
  AssertingVH<CallInst> SetShape1 = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape1, Idx0Val}); // [3]

  // Second tensor setup: [7] - incompatible with [3] for broadcasting
  AssertingVH<CallInst> SetShape2 = IRB.CreateCall(
      SetShapeFn, {PE, Dim1, One, One, One, One, One, One, One, One, One});
  Value *Idx1Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index1 =
      IRB.CreateCall(IndexFn, {SetShape2, Idx1Val}); // [7]

  Type *Int64Ty = IRB.getInt64Ty();
  Type *PtrTy = PointerType::get(C, 0);

  // Create tensor addresses
  AssertingVH<Value> TensorAddr3 =
      IRB.CreateAdd(Index0, IRB.getInt64(2), "tensor_addr3"); // [3]
  AssertingVH<Value> TensorAddrPtr3 =
      IRB.CreateIntToPtr(TensorAddr3, PtrTy, "tensor_addr_ptr3");

  // Test: Try to create GEP with incompatible tensor shapes
  // [3] address + [7] index should fail because 3 != 7 in the first dimension
  // Neither dimension is 1, so they cannot be broadcasted together
  IRB.CreateGEP(Int64Ty, TensorAddrPtr3, {Index1}, "incompatible_gep");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);

  // Expect shape propagation to fail due to incompatible broadcast shapes in
  // GEP
  ASSERT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));

  // Restore original diagnostic handler
  if (OriginalHandler) {
    C.setDiagnosticHandler(std::move(OriginalHandler));
  }
}

TEST_F(PropagateShapeTest, BranchCondition) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);

  // Create basic blocks: Entry -> A | Exit (tensor condition)
  //                            A -> B | Exit (scalar condition)
  //                            B -> Exit
  BasicBlock *EntryBB = BasicBlock::Create(C, "Entry", F);
  BasicBlock *ABB = BasicBlock::Create(C, "A", F);
  BasicBlock *BBB = BasicBlock::Create(C, "B", F);
  BasicBlock *ExitBB = BasicBlock::Create(C, "Exit", F);

  Function *SetShapeFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_setshape, {IRB.getInt64Ty()});
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  // Entry Block: Create tensor condition and branch
  IRB.SetInsertPoint(EntryBB);
  Value *PE = IRB.getInt64(0);
  Value *Dim0 = IRB.getInt64(4);
  Value *One = IRB.getInt64(1);
  AssertingVH<CallInst> SetShape = IRB.CreateCall(
      SetShapeFn, {PE, Dim0, One, One, One, One, One, One, One, One, One});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index0 =
      IRB.CreateCall(IndexFn, {SetShape, Idx0Val}); // [4]

  Value *Scalar5 = IRB.getInt64(5);
  Value *Scalar10 = IRB.getInt64(10);
  Value *Scalar20 = IRB.getInt64(20);

  // Tensor condition - tensor > scalar -> tensor boolean [4]
  AssertingVH<Value> TensorCondition = IRB.CreateICmpSGT(
      Index0, Scalar5, "tensor_cond"); // [4] > 5 -> [4] boolean

  // Branch on tensor condition (element-wise branching)
  IRB.CreateCondBr(TensorCondition, ABB, ExitBB);

  // Block A: Create scalar condition and branch again
  IRB.SetInsertPoint(ABB);
  AssertingVH<Value> AResult =
      IRB.CreateAdd(Index0, Scalar10, "a_result"); // [4] + scalar -> [4]

  // Scalar condition - scalar > scalar -> scalar boolean
  AssertingVH<Value> ScalarCondition =
      IRB.CreateICmpSGT(Scalar20, Scalar10, "scalar_cond"); // 20 > 10 -> true

  // Branch on scalar condition
  IRB.CreateCondBr(ScalarCondition, BBB, ExitBB);

  // Block B: Final operations before exit
  IRB.SetInsertPoint(BBB);
  AssertingVH<Value> BResult = IRB.CreateMul(AResult, IRB.getInt64(2),
                                             "b_result"); // [4] * scalar -> [4]
  IRB.CreateBr(ExitBB);

  // Exit Block: Return void
  IRB.SetInsertPoint(ExitBB);
  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify tensor condition: tensor comparison -> tensor (same shape as input)
  const TensorShape &TensorCondShape = Ripple->getRippleShape(TensorCondition);
  EXPECT_EQ(TensorCondShape.rank(), 1u);
  EXPECT_EQ(TensorCondShape[0], 4u);

  // Verify scalar condition: scalar comparison -> scalar
  const TensorShape &ScalarCondShape = Ripple->getRippleShape(ScalarCondition);
  EXPECT_TRUE(ScalarCondShape.isScalar());

  // Verify operations in branches
  const TensorShape &AResultShape = Ripple->getRippleShape(AResult);
  EXPECT_EQ(AResultShape.rank(), 1u);
  EXPECT_EQ(AResultShape[0], 4u);

  const TensorShape &BResultShape = Ripple->getRippleShape(BResult);
  EXPECT_EQ(BResultShape.rank(), 1u);
  EXPECT_EQ(BResultShape[0], 4u);
}

TEST_F(PropagateShapeTest, RippleBroadcast) {
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
  Function *BroadcastFloatFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_broadcast, {IRB.getFloatTy()});

  // Create different target shapes for broadcasting
  Value *PE = IRB.getInt64(0);
  Value *One = IRB.getInt64(1);

  // Target shape 1: [4] (1D)
  Value *Dim4 = IRB.getInt64(4);
  AssertingVH<CallInst> SetShape1D = IRB.CreateCall(
      SetShapeFn, {PE, Dim4, One, One, One, One, One, One, One, One, One});

  // Target shape 2: [3][5] (2D)
  Value *Dim3 = IRB.getInt64(3);
  Value *Dim5 = IRB.getInt64(5);
  AssertingVH<CallInst> SetShape2D = IRB.CreateCall(
      SetShapeFn, {PE, Dim3, Dim5, One, One, One, One, One, One, One, One});

  // Target shape 3: [6][2] (2D different dimensions)
  Value *Dim6 = IRB.getInt64(6);
  Value *Dim2 = IRB.getInt64(2);
  AssertingVH<CallInst> SetShape2D_Alt = IRB.CreateCall(
      SetShapeFn, {PE, Dim6, Dim2, One, One, One, One, One, One, One, One});

  // Create scalar values for broadcasting
  Value *ScalarInt = IRB.getInt64(42);
  Value *ScalarFloat = ConstantFP::get(IRB.getFloatTy(), 3.14f);

  // Test 1: Scalar int -> 1D broadcast (bitset 0x1 = dimension 0)
  Value *Bitset1D = IRB.getInt64(0x1); // Broadcast along dimension 0
  AssertingVH<CallInst> Broadcast1 = IRB.CreateCall(
      BroadcastFn, {SetShape1D, Bitset1D, ScalarInt}, "broadcast_scalar_to_1d");

  // Test 2: Scalar int -> 2D broadcast (bitset 0x3 = dimensions 0 and 1)
  Value *Bitset2D = IRB.getInt64(0x3); // Broadcast along dimensions 0 and 1
  AssertingVH<CallInst> Broadcast2 = IRB.CreateCall(
      BroadcastFn, {SetShape2D, Bitset2D, ScalarInt}, "broadcast_scalar_to_2d");

  // Test 3: Scalar float -> 1D broadcast
  AssertingVH<CallInst> Broadcast3 =
      IRB.CreateCall(BroadcastFloatFn, {SetShape1D, Bitset1D, ScalarFloat},
                     "broadcast_scalar_float_to_1d");

  // Test 4: Scalar float -> 2D broadcast
  AssertingVH<CallInst> Broadcast4 =
      IRB.CreateCall(BroadcastFloatFn, {SetShape2D, Bitset2D, ScalarFloat},
                     "broadcast_scalar_float_to_2d");

  // Test 5: Scalar int -> 2D broadcast with different dimensions
  AssertingVH<CallInst> Broadcast5 =
      IRB.CreateCall(BroadcastFn, {SetShape2D_Alt, Bitset2D, ScalarInt},
                     "broadcast_scalar_to_2d_alt");

  // For 1D -> 2D broadcast, we need to create a 1D tensor first
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index1D =
      IRB.CreateCall(IndexFn, {SetShape1D, Idx0Val}); // Creates [4] tensor

  // Test 6: 1D int -> 2D broadcast (bitset 0x2 = dimension 1 only)
  Value *BitsetDim1 = IRB.getInt64(0x2); // Broadcast along dimension 1 only
  AssertingVH<CallInst> Broadcast6 = IRB.CreateCall(
      BroadcastFn, {SetShape2D, BitsetDim1, Index1D}, "broadcast_1d_to_2d");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify Test 1: Scalar int -> 1D [4]
  const TensorShape &Broadcast1Shape =
      Ripple->getRippleShape(static_cast<Value *>(Broadcast1));
  EXPECT_EQ(Broadcast1Shape.rank(), 2u);
  EXPECT_EQ(Broadcast1Shape[0], 4u);
  EXPECT_EQ(Broadcast1Shape[1], 1u);

  // Verify Test 2: Scalar int -> 2D [3][5]
  const TensorShape &Broadcast2Shape =
      Ripple->getRippleShape(static_cast<Value *>(Broadcast2));
  EXPECT_EQ(Broadcast2Shape.rank(), 2u);
  EXPECT_EQ(Broadcast2Shape[0], 3u);
  EXPECT_EQ(Broadcast2Shape[1], 5u);

  // Verify Test 3: Scalar float -> 1D [4]
  const TensorShape &Broadcast3Shape =
      Ripple->getRippleShape(static_cast<Value *>(Broadcast3));
  EXPECT_EQ(Broadcast3Shape.rank(), 2u);
  EXPECT_EQ(Broadcast3Shape[0], 4u);
  EXPECT_EQ(Broadcast3Shape[1], 1u);

  // Verify Test 4: Scalar float -> 2D [3][5]
  const TensorShape &Broadcast4Shape =
      Ripple->getRippleShape(static_cast<Value *>(Broadcast4));
  EXPECT_EQ(Broadcast4Shape.rank(), 2u);
  EXPECT_EQ(Broadcast4Shape[0], 3u);
  EXPECT_EQ(Broadcast4Shape[1], 5u);

  // Verify Test 5: Scalar int -> 2D [6][2]
  const TensorShape &Broadcast5Shape =
      Ripple->getRippleShape(static_cast<Value *>(Broadcast5));
  EXPECT_EQ(Broadcast5Shape.rank(), 2u);
  EXPECT_EQ(Broadcast5Shape[0], 6u);
  EXPECT_EQ(Broadcast5Shape[1], 2u);

  // Verify Test 6: 1D [4] bcast 2D [3][5] (broadcast along dimension 1)
  // Input [4] should be broadcast to [4][5]

  // TODO: that's currently underspecified and seems wrong, maybe this should be
  // an error!
  const TensorShape &Broadcast6Shape =
      Ripple->getRippleShape(static_cast<Value *>(Broadcast6));
  EXPECT_EQ(Broadcast6Shape.rank(), 2u);
  EXPECT_EQ(Broadcast6Shape[0], 4u);
  EXPECT_EQ(Broadcast6Shape[1], 5u);

  // Verify the original 1D tensor shape
  const TensorShape &Index1DShape =
      Ripple->getRippleShape(static_cast<Value *>(Index1D));
  EXPECT_EQ(Index1DShape.rank(), 2u);
  EXPECT_EQ(Index1DShape[0], 4u);
  EXPECT_EQ(Index1DShape[1], 1u);
}

TEST_F(PropagateShapeTest, RippleBroadcastFail) {
  // Set up diagnostic handler to suppress diagnostics output to avoid polluting
  // the test stderr for the expected error
  auto OriginalHandler = C.getDiagnosticHandler();
  C.setDiagnosticHandler(std::make_unique<SuppressingDiagnosticHandler>());

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

  // Create target shape for broadcasting: [6][2]
  Value *PE = IRB.getInt64(0);
  Value *One = IRB.getInt64(1);
  Value *Dim6 = IRB.getInt64(6);
  Value *Dim2 = IRB.getInt64(2);
  AssertingVH<CallInst> SetShape2D_Alt = IRB.CreateCall(
      SetShapeFn, {PE, Dim6, Dim2, One, One, One, One, One, One, One, One});

  // Create a 1D tensor [4] for broadcasting
  Value *Dim4 = IRB.getInt64(4);
  AssertingVH<CallInst> SetShape1D = IRB.CreateCall(
      SetShapeFn, {PE, Dim4, One, One, One, One, One, One, One, One, One});

  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index1D =
      IRB.CreateCall(IndexFn, {SetShape1D, Idx0Val}); // Creates [4] tensor

  // Test: Try to broadcast [4] to [6][2] (bitset 0x1 = dimension 0 only)
  // This should fail because [4] (i.e., [4][1]) cannot be broadcasted to [6][2]
  // The shape [4] in dimension 0 is incompatible with target shape [6] in
  // dimension 0
  Value *BitsetDim0 = IRB.getInt64(0x1); // Broadcast along dimension 0 only
  IRB.CreateCall(BroadcastFn, {SetShape2D_Alt, BitsetDim0, Index1D},
                 "broadcast_1d_to_2d_fail");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);

  // Expect shape propagation to fail due to incompatible broadcast shapes
  // [4] (i.e., [4][1]) cannot be broadcasted to [6][2]
  ASSERT_TRUE(static_cast<bool>(Err));
  consumeError(std::move(Err));

  // Restore original diagnostic handler
  if (OriginalHandler) {
    C.setDiagnosticHandler(std::move(OriginalHandler));
  }
}

TEST_F(PropagateShapeTest, RippleReduce) {
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
  Function *ReduceAddFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_reduce_add, {IRB.getInt64Ty()});
  Function *ReduceFAddFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_reduce_fadd, {IRB.getFloatTy()});

  // Create different source shapes for reduction
  Value *PE = IRB.getInt64(0);
  Value *One = IRB.getInt64(1);

  // Source shape 1: [4][6] (2D)
  Value *Dim4 = IRB.getInt64(4);
  Value *Dim6 = IRB.getInt64(6);
  AssertingVH<CallInst> SetShape2D = IRB.CreateCall(
      SetShapeFn, {PE, Dim4, Dim6, One, One, One, One, One, One, One, One});

  // Source shape 2: [8] (1D)
  Value *Dim8 = IRB.getInt64(8);
  AssertingVH<CallInst> SetShape1D = IRB.CreateCall(
      SetShapeFn, {PE, Dim8, One, One, One, One, One, One, One, One, One});

  // Create tensors for reduction
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index2D_Dim0 = IRB.CreateCall(
      IndexFn, {SetShape2D, Idx0Val}); // Creates [4] tensor from [4][6] shape

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index2D_Dim1 = IRB.CreateCall(
      IndexFn,
      {SetShape2D, Idx1Val}); // Creates [1][6] tensor from [4][6] shape

  AssertingVH<CallInst> Index1D =
      IRB.CreateCall(IndexFn, {SetShape1D, Idx0Val}); // Creates [8] tensor

  // Create float tensors for float reduction
  AssertingVH<Value> FloatTensor2D =
      IRB.CreateSIToFP(Index2D_Dim0, IRB.getFloatTy(), "float_tensor_2d");
  AssertingVH<Value> FloatTensor1D =
      IRB.CreateSIToFP(Index1D, IRB.getFloatTy(), "float_tensor_1d");

  // Test 1: 2D -> 1D reduction (reduce along dimension 0)
  // [4][6] -> [1][6] by reducing dimension 0 (bitset 0x1)
  Value *BitsetDim0 = IRB.getInt64(0x1); // Reduce along dimension 0
  AssertingVH<CallInst> Reduce1 = IRB.CreateCall(
      ReduceAddFn, {BitsetDim0, Index2D_Dim0}, "reduce_2d_to_1d_dim0");

  // Test 2: 2D -> 1D reduction (reduce along dimension 1)
  // [4][6] -> [4][1] by reducing dimension 1 (bitset 0x2)
  Value *BitsetDim1 = IRB.getInt64(0x2); // Reduce along dimension 1
  AssertingVH<CallInst> Reduce2 = IRB.CreateCall(
      ReduceAddFn, {BitsetDim1, Index2D_Dim1}, "reduce_2d_to_1d_dim1");

  // Test 3: 2D -> scalar reduction (reduce along both dimensions)
  // [4][6] -> scalar by reducing dimensions 0 and 1 (bitset 0x3)
  Value *BitsetBoth = IRB.getInt64(0x3); // Reduce along dimensions 0 and 1
  AssertingVH<CallInst> Reduce3 = IRB.CreateCall(
      ReduceAddFn, {BitsetBoth, Index2D_Dim0}, "reduce_2d_to_scalar");

  // Test 4: 1D -> scalar reduction
  // [8] -> scalar by reducing dimension 0 (bitset 0x1)
  AssertingVH<CallInst> Reduce4 =
      IRB.CreateCall(ReduceAddFn, {BitsetDim0, Index1D}, "reduce_1d_to_scalar");

  // Test 5: Float 2D -> 1D reduction
  // [4][6] -> [1][6] by reducing dimension 0 (bitset 0x1)
  AssertingVH<CallInst> Reduce5 = IRB.CreateCall(
      ReduceFAddFn, {BitsetDim0, FloatTensor2D}, "reduce_float_2d_to_1d");

  // Test 6: Float 1D -> scalar reduction
  // [8] -> scalar by reducing dimension 0 (bitset 0x1)
  AssertingVH<CallInst> Reduce6 = IRB.CreateCall(
      ReduceFAddFn, {BitsetDim0, FloatTensor1D}, "reduce_float_1d_to_scalar");

  // Test 7: Scalar -> scalar reduction (should remain scalar)
  // scalar -> scalar by reducing dimension 0 (bitset 0x1)
  Value *ScalarInt = IRB.getInt64(42);
  AssertingVH<CallInst> Reduce7 = IRB.CreateCall(
      ReduceAddFn, {BitsetDim0, ScalarInt}, "reduce_scalar_to_scalar");

  // Test 8: Float scalar -> scalar reduction (should remain scalar)
  Value *ScalarFloat = ConstantFP::get(IRB.getFloatTy(), 3.14f);
  AssertingVH<CallInst> Reduce8 = IRB.CreateCall(
      ReduceFAddFn, {BitsetBoth, ScalarFloat}, "reduce_float_scalar_to_scalar");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify Test 1: 2D [4][6] -> 1D [1][6] (reduce dimension 0)
  const TensorShape &Reduce1Shape =
      Ripple->getRippleShape(static_cast<Value *>(Reduce1));
  EXPECT_EQ(Reduce1Shape.rank(), 2u);
  EXPECT_EQ(Reduce1Shape[0], 1u);
  EXPECT_EQ(Reduce1Shape[1], 1u); // Original [4] becomes [1] after reduction

  // Verify Test 2: 2D [4][6] -> 1D [4][1] (reduce dimension 1)
  const TensorShape &Reduce2Shape =
      Ripple->getRippleShape(static_cast<Value *>(Reduce2));
  EXPECT_EQ(Reduce2Shape.rank(), 2u);
  EXPECT_EQ(Reduce2Shape[0], 1u);
  EXPECT_EQ(Reduce2Shape[1],
            1u); // Original [1][6] becomes [1][1] after reduction

  // Verify Test 3: 2D [4][6] -> scalar (reduce both dimensions)
  const TensorShape &Reduce3Shape =
      Ripple->getRippleShape(static_cast<Value *>(Reduce3));
  EXPECT_TRUE(Reduce3Shape.isScalar());

  // Verify Test 4: 1D [8] -> scalar (reduce dimension 0)
  const TensorShape &Reduce4Shape =
      Ripple->getRippleShape(static_cast<Value *>(Reduce4));
  EXPECT_TRUE(Reduce4Shape.isScalar());

  // Verify Test 5: Float 2D [4][6] -> 1D [1][6] (reduce dimension 0)
  const TensorShape &Reduce5Shape =
      Ripple->getRippleShape(static_cast<Value *>(Reduce5));
  EXPECT_EQ(Reduce5Shape.rank(), 2u);
  EXPECT_EQ(Reduce5Shape[0], 1u);
  EXPECT_EQ(Reduce5Shape[1], 1u);

  // Verify Test 6: Float 1D [8] -> scalar (reduce dimension 0)
  const TensorShape &Reduce6Shape =
      Ripple->getRippleShape(static_cast<Value *>(Reduce6));
  EXPECT_TRUE(Reduce6Shape.isScalar());

  // Verify Test 7: Scalar -> scalar (reduce dimension 0)
  const TensorShape &Reduce7Shape =
      Ripple->getRippleShape(static_cast<Value *>(Reduce7));
  EXPECT_TRUE(Reduce7Shape.isScalar());

  // Verify Test 8: Float scalar -> scalar (reduce both dimensions)
  const TensorShape &Reduce8Shape =
      Ripple->getRippleShape(static_cast<Value *>(Reduce8));
  EXPECT_TRUE(Reduce8Shape.isScalar());

  // Verify the original tensor shapes
  const TensorShape &Index2D_Dim0Shape =
      Ripple->getRippleShape(static_cast<Value *>(Index2D_Dim0));
  EXPECT_EQ(Index2D_Dim0Shape.rank(), 2u);
  EXPECT_EQ(Index2D_Dim0Shape[0], 4u);
  EXPECT_EQ(Index2D_Dim0Shape[1], 1u);

  const TensorShape &Index2D_Dim1Shape =
      Ripple->getRippleShape(static_cast<Value *>(Index2D_Dim1));
  EXPECT_EQ(Index2D_Dim1Shape.rank(), 2u);
  EXPECT_EQ(Index2D_Dim1Shape[0], 1u);
  EXPECT_EQ(Index2D_Dim1Shape[1], 6u);

  const TensorShape &Index1DShape =
      Ripple->getRippleShape(static_cast<Value *>(Index1D));
  EXPECT_EQ(Index1DShape.rank(), 2u);
  EXPECT_EQ(Index1DShape[0], 8u);
  EXPECT_EQ(Index1DShape[1], 1u);
}

TEST_F(PropagateShapeTest, RippleSlice) {
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
  Function *SliceFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_slice, {IRB.getInt64Ty()});
  Function *SliceFloatFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_slice, {IRB.getFloatTy()});

  // Create different source shapes for slicing
  Value *PE = IRB.getInt64(0);
  Value *One = IRB.getInt64(1);

  // Source shape 1: [4][6][8] (3D)
  Value *Dim4 = IRB.getInt64(4);
  Value *Dim6 = IRB.getInt64(6);
  Value *Dim8 = IRB.getInt64(8);
  AssertingVH<CallInst> SetShape3D = IRB.CreateCall(
      SetShapeFn, {PE, Dim4, Dim6, Dim8, One, One, One, One, One, One, One});

  // Source shape 2: [10][5] (2D)
  Value *Dim10 = IRB.getInt64(10);
  Value *Dim5 = IRB.getInt64(5);
  AssertingVH<CallInst> SetShape2D = IRB.CreateCall(
      SetShapeFn, {PE, Dim10, Dim5, One, One, One, One, One, One, One, One});

  // Create tensors for slicing
  Function *IndexFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::ripple_block_index, {IRB.getInt64Ty()});

  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index3D_Dim0 = IRB.CreateCall(
      IndexFn,
      {SetShape3D, Idx0Val}); // Creates [4] tensor from [4][6][8] shape

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index3D_Dim1 = IRB.CreateCall(
      IndexFn,
      {SetShape3D, Idx1Val}); // Creates [1][6] tensor from [4][6][8] shape

  Value *Idx2Val = IRB.getInt64(2);
  AssertingVH<CallInst> Index3D_Dim2 = IRB.CreateCall(
      IndexFn,
      {SetShape3D, Idx2Val}); // Creates [1][1][8] tensor from [4][6][8] shape

  AssertingVH<CallInst> Index2D_Dim0 = IRB.CreateCall(
      IndexFn, {SetShape2D, Idx0Val}); // Creates [10] tensor from [10][5] shape

  AssertingVH<CallInst> Index2D_Dim1 = IRB.CreateCall(
      IndexFn,
      {SetShape2D, Idx1Val}); // Creates [1][5] tensor from [10][5] shape

  // Create float tensors for float slicing
  AssertingVH<Value> FloatTensor3D =
      IRB.CreateSIToFP(Index3D_Dim0, IRB.getFloatTy(), "float_tensor_3d");
  AssertingVH<Value> FloatTensor2D =
      IRB.CreateSIToFP(Index2D_Dim0, IRB.getFloatTy(), "float_tensor_2d");

  // Slice parameters: positive indices select specific elements, -1 keeps
  // dimension
  Value *KeepDim = IRB.getInt64(-1);  // Keep dimension unchanged
  Value *SliceIdx0 = IRB.getInt64(0); // Select element 0
  Value *SliceIdx2 = IRB.getInt64(2); // Select element 2
  Value *SliceIdx3 = IRB.getInt64(3); // Select element 3

  // Test 1: 3D -> 2D slice (slice dimension 0 at index 2)
  // [4][6][8] -> [1][6][8] by slicing dimension 0 at index 2
  AssertingVH<CallInst> Slice1 =
      IRB.CreateCall(SliceFn,
                     {Index3D_Dim0, SliceIdx2, KeepDim, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_3d_dim0");

  // Test 2: 3D -> 2D slice (slice dimension 1 at index 3)
  // [4][6][8] -> [4][1][8] by slicing dimension 1 at index 3
  AssertingVH<CallInst> Slice2 =
      IRB.CreateCall(SliceFn,
                     {Index3D_Dim1, KeepDim, SliceIdx3, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_3d_dim1");

  // Test 3: 3D -> 2D slice (slice dimension 2 at index 0)
  // [4][6][8] -> [4][6][1] by slicing dimension 2 at index 0
  AssertingVH<CallInst> Slice3 =
      IRB.CreateCall(SliceFn,
                     {Index3D_Dim2, KeepDim, KeepDim, SliceIdx0, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_3d_dim2");

  // Test 4: 3D -> 1D slice (slice dimensions 0 and 1)
  // [4][6][8] -> [1][1][8] by slicing dimensions 0 at index 1 and 1 at index 2
  Value *SliceIdx1 = IRB.getInt64(1);
  AssertingVH<CallInst> Slice4 =
      IRB.CreateCall(SliceFn,
                     {Index3D_Dim0, SliceIdx1, SliceIdx2, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_3d_two_dims");

  // Test 5: 3D -> scalar slice (slice all dimensions)
  // [4][6][8] -> scalar by slicing all dimensions
  AssertingVH<CallInst> Slice5 =
      IRB.CreateCall(SliceFn,
                     {Index3D_Dim0, SliceIdx0, SliceIdx1, SliceIdx2, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_3d_to_scalar");

  // Test 6: 2D -> 1D slice (slice dimension 0)
  // [10][5] -> [1][5] by slicing dimension 0 at index 3
  AssertingVH<CallInst> Slice6 =
      IRB.CreateCall(SliceFn,
                     {Index2D_Dim0, SliceIdx3, KeepDim, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_2d_dim0");

  // Test 7: 2D -> 1D slice (slice dimension 1)
  // [10][5] -> [10][1] by slicing dimension 1 at index 2
  AssertingVH<CallInst> Slice7 =
      IRB.CreateCall(SliceFn,
                     {Index2D_Dim1, KeepDim, SliceIdx2, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_2d_dim1");

  // Test 8: 2D -> scalar slice (slice both dimensions)
  // [10][5] -> scalar by slicing both dimensions
  AssertingVH<CallInst> Slice8 =
      IRB.CreateCall(SliceFn,
                     {Index2D_Dim0, SliceIdx2, SliceIdx1, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_2d_to_scalar");

  // Test 9: Float 3D -> 2D slice
  // [4][6][8] -> [1][6][8] by slicing dimension 0 at index 1
  AssertingVH<CallInst> Slice9 =
      IRB.CreateCall(SliceFloatFn,
                     {FloatTensor3D, SliceIdx1, KeepDim, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_float_3d");

  // Test 10: Float 2D -> scalar slice
  // [10][5] -> scalar by slicing both dimensions
  AssertingVH<CallInst> Slice10 =
      IRB.CreateCall(SliceFloatFn,
                     {FloatTensor2D, SliceIdx0, SliceIdx3, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_float_2d_to_scalar");

  // Test 11: Scalar -> scalar slice (should remain scalar)
  Value *ScalarInt = IRB.getInt64(42);
  AssertingVH<CallInst> Slice11 =
      IRB.CreateCall(SliceFn,
                     {ScalarInt, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_scalar");

  // Test 12: Float scalar -> scalar slice (should remain scalar)
  Value *ScalarFloat = ConstantFP::get(IRB.getFloatTy(), 3.14f);
  AssertingVH<CallInst> Slice12 =
      IRB.CreateCall(SliceFloatFn,
                     {ScalarFloat, KeepDim, KeepDim, KeepDim, KeepDim, KeepDim,
                      KeepDim, KeepDim, KeepDim, KeepDim, KeepDim},
                     "slice_float_scalar");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify Test 1: 3D [4][6][8] -> [1][6][8] (slice dimension 0)
  const TensorShape &Slice1Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice1));
  EXPECT_EQ(Slice1Shape[0], 1u); // Sliced dimension becomes 1
  EXPECT_EQ(Slice1Shape[1], 1u); // Original [4] becomes [1] after slicing

  // Verify Test 2: 3D [4][6][8] -> [4][1][8] (slice dimension 1)
  const TensorShape &Slice2Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice2));
  EXPECT_EQ(Slice2Shape[0],
            1u); // Original [1][6] becomes [1][1] after slicing dim 1
  EXPECT_EQ(Slice2Shape[1], 1u);

  // Verify Test 3: 3D [4][6][8] -> [4][6][1] (slice dimension 2)
  const TensorShape &Slice3Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice3));
  EXPECT_EQ(Slice3Shape[0],
            1u); // Original [1][1][8] becomes [1][1][1] after slicing dim 2
  EXPECT_EQ(Slice3Shape[1], 1u);

  // Verify Test 4: 3D [4][6][8] -> [1][1][8] (slice dimensions 0 and 1)
  const TensorShape &Slice4Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice4));
  EXPECT_EQ(Slice4Shape[0], 1u); // Both sliced dimensions become 1
  EXPECT_EQ(Slice4Shape[1], 1u);

  // Verify Test 5: 3D [4][6][8] -> scalar (slice all dimensions)
  const TensorShape &Slice5Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice5));
  EXPECT_TRUE(Slice5Shape.isScalar());

  // Verify Test 6: 2D [10][5] -> [1][5] (slice dimension 0)
  const TensorShape &Slice6Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice6));
  EXPECT_EQ(Slice6Shape[0], 1u); // Sliced dimension becomes 1
  EXPECT_EQ(Slice6Shape[1], 1u); // Original [10] becomes [1] after slicing

  // Verify Test 7: 2D [10][5] -> [10][1] (slice dimension 1)
  const TensorShape &Slice7Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice7));
  EXPECT_EQ(Slice7Shape[0],
            1u); // Original [1][5] becomes [1][1] after slicing dim 1
  EXPECT_EQ(Slice7Shape[1], 1u);

  // Verify Test 8: 2D [10][5] -> scalar (slice both dimensions)
  const TensorShape &Slice8Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice8));
  EXPECT_TRUE(Slice8Shape.isScalar());

  // Verify Test 9: Float 3D [4][6][8] -> [1][6][8] (slice dimension 0)
  const TensorShape &Slice9Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice9));
  EXPECT_EQ(Slice9Shape[0], 1u);
  EXPECT_EQ(Slice9Shape[1], 1u);

  // Verify Test 10: Float 2D [10][5] -> scalar (slice both dimensions)
  const TensorShape &Slice10Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice10));
  EXPECT_TRUE(Slice10Shape.isScalar());

  // Verify Test 11: Scalar -> scalar (slice with all keep dims)
  const TensorShape &Slice11Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice11));
  EXPECT_TRUE(Slice11Shape.isScalar());

  // Verify Test 12: Float scalar -> scalar (slice with all keep dims)
  const TensorShape &Slice12Shape =
      Ripple->getRippleShape(static_cast<Value *>(Slice12));
  EXPECT_TRUE(Slice12Shape.isScalar());

  // Verify the original tensor shapes
  const TensorShape &Index3D_Dim0Shape =
      Ripple->getRippleShape(static_cast<Value *>(Index3D_Dim0));
  EXPECT_EQ(Index3D_Dim0Shape.rank(), 3u);
  EXPECT_EQ(Index3D_Dim0Shape[0], 4u);
  EXPECT_EQ(Index3D_Dim0Shape[1], 1u);

  const TensorShape &Index3D_Dim1Shape =
      Ripple->getRippleShape(static_cast<Value *>(Index3D_Dim1));
  EXPECT_EQ(Index3D_Dim1Shape[0], 1u);
  EXPECT_EQ(Index3D_Dim1Shape[1], 6u);

  const TensorShape &Index2D_Dim0Shape =
      Ripple->getRippleShape(static_cast<Value *>(Index2D_Dim0));
  EXPECT_EQ(Index2D_Dim0Shape[0], 10u);
  EXPECT_EQ(Index2D_Dim0Shape[1], 1u);

  const TensorShape &Index2D_Dim1Shape =
      Ripple->getRippleShape(static_cast<Value *>(Index2D_Dim1));
  EXPECT_EQ(Index2D_Dim1Shape[0], 1u);
  EXPECT_EQ(Index2D_Dim1Shape[1], 5u);
}

TEST_F(PropagateShapeTest, AllocaStaysScalar) {
  IRBuilder<> IRB(C);
  Module M("TestModule", C);
  Type *VoidTy = IRB.getVoidTy();
  FunctionType *FTy = FunctionType::get(VoidTy, /*isVarArg*/ false);
  Function *F = Function::Create(
      FTy, GlobalValue::LinkageTypes::ExternalLinkage, "TestF", M);
  BasicBlock *BB = BasicBlock::Create(C, "EntryBB", F);
  IRB.SetInsertPoint(BB);

  // Create scalar values
  Value *Scalar10 = IRB.getInt64(10);
  Value *Scalar20 = IRB.getInt64(20);
  Value *Scalar5 = IRB.getInt64(5);

  // Test: Alloca for scalar operations only
  Type *Int64Ty = IRB.getInt64Ty();
  AssertingVH<Value> AllocaScalar =
      IRB.CreateAlloca(Int64Ty, nullptr, "alloca_scalar");

  // Store scalar to alloca
  IRB.CreateStore(Scalar10, AllocaScalar);

  // Load scalar from alloca
  AssertingVH<Value> LoadScalar1 =
      IRB.CreateLoad(Int64Ty, AllocaScalar, "load_scalar1");

  // Perform scalar operation: scalar + scalar -> scalar
  AssertingVH<Value> AddScalar =
      IRB.CreateAdd(LoadScalar1, Scalar20, "add_scalar");

  // Store scalar result back
  IRB.CreateStore(AddScalar, AllocaScalar);

  // Load again and perform another scalar operation
  AssertingVH<Value> LoadScalar2 =
      IRB.CreateLoad(Int64Ty, AllocaScalar, "load_scalar2");
  AssertingVH<Value> MulScalar =
      IRB.CreateMul(LoadScalar2, Scalar5, "mul_scalar");

  // Store final result
  IRB.CreateStore(MulScalar, AllocaScalar);

  // Final load to verify
  AssertingVH<Value> LoadScalarFinal =
      IRB.CreateLoad(Int64Ty, AllocaScalar, "load_scalar_final");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify all operations remain scalar
  const TensorShape &AllocaScalarShape = Ripple->getRippleShape(AllocaScalar);
  EXPECT_TRUE(AllocaScalarShape.isScalar());

  const TensorShape &LoadScalar1Shape = Ripple->getRippleShape(LoadScalar1);
  EXPECT_TRUE(LoadScalar1Shape.isScalar());

  const TensorShape &AddScalarShape = Ripple->getRippleShape(AddScalar);
  EXPECT_TRUE(AddScalarShape.isScalar());

  const TensorShape &LoadScalar2Shape = Ripple->getRippleShape(LoadScalar2);
  EXPECT_TRUE(LoadScalar2Shape.isScalar());

  const TensorShape &MulScalarShape = Ripple->getRippleShape(MulScalar);
  EXPECT_TRUE(MulScalarShape.isScalar());

  const TensorShape &LoadScalarFinalShape =
      Ripple->getRippleShape(LoadScalarFinal);
  EXPECT_TRUE(LoadScalarFinalShape.isScalar());
}

TEST_F(PropagateShapeTest, AllocaPromotion1Dto2D) {
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

  // Create tensor shapes for testing: [8][4] (2D)
  Value *PE = IRB.getInt64(0);
  Value *One = IRB.getInt64(1);
  Value *Dim8 = IRB.getInt64(8);
  Value *Dim4 = IRB.getInt64(4);
  AssertingVH<CallInst> SetShape2D = IRB.CreateCall(
      SetShapeFn, {PE, Dim8, Dim4, One, One, One, One, One, One, One, One});

  // Create tensors for testing
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index2D_Dim0 = IRB.CreateCall(
      IndexFn, {SetShape2D, Idx0Val}); // Creates [8] tensor from [8][4] shape

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index2D_Dim1 = IRB.CreateCall(
      IndexFn,
      {SetShape2D, Idx1Val}); // Creates [1][4] tensor from [8][4] shape

  // Create scalar value
  Value *Scalar10 = IRB.getInt64(10);

  // Test: Alloca that starts with [8] and becomes [8][4] through broadcasting
  Type *Int64Ty = IRB.getInt64Ty();
  AssertingVH<Value> Alloca2D = IRB.CreateAlloca(Int64Ty, nullptr, "alloca_2d");

  // Store initial [8] tensor to alloca
  IRB.CreateStore(Index2D_Dim0, Alloca2D);

  // Load the [8] tensor from alloca
  AssertingVH<Value> Load2D_1 = IRB.CreateLoad(Int64Ty, Alloca2D, "load_2d_1");

  // Perform binary operation that causes broadcasting: [8] + [1][4] -> [8][4]
  AssertingVH<Value> Add2D = IRB.CreateAdd(Load2D_1, Index2D_Dim1, "add_2d");

  // Store the broadcast result [8][4] back to alloca
  IRB.CreateStore(Add2D, Alloca2D);

  // Load again - should now have [8][4] shape
  AssertingVH<Value> Load2D_2 = IRB.CreateLoad(Int64Ty, Alloca2D, "load_2d_2");

  // Perform scalar operation: [8][4] - scalar -> [8][4]
  AssertingVH<Value> Sub2D = IRB.CreateSub(Load2D_2, Scalar10, "sub_2d");

  // Store final result
  IRB.CreateStore(Sub2D, Alloca2D);

  // Final load to verify
  AssertingVH<Value> Load2D_Final =
      IRB.CreateLoad(Int64Ty, Alloca2D, "load_2d_final");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify shape progression: [8] -> [8][4]
  // Alloca should end up with [8][4] shape (large enough for both [8] and
  // [8][4] shapes)
  const TensorShape &Alloca2DShape = Ripple->getRippleShape(Alloca2D);
  EXPECT_EQ(Alloca2DShape.rank(), 2u);
  EXPECT_EQ(Alloca2DShape[0], 8u);
  EXPECT_EQ(Alloca2DShape[1], 4u);

  // First load should have original [8] shape
  const TensorShape &Load2D_1Shape = Ripple->getRippleShape(Load2D_1);
  EXPECT_EQ(Load2D_1Shape.rank(), 2u);
  EXPECT_EQ(Load2D_1Shape[0], 8u);
  EXPECT_EQ(Load2D_1Shape[1], 1u);

  // Binary operation should result in broadcast [8][4] shape
  const TensorShape &Add2DShape = Ripple->getRippleShape(Add2D);
  EXPECT_EQ(Add2DShape.rank(), 2u);
  EXPECT_EQ(Add2DShape[0], 8u);
  EXPECT_EQ(Add2DShape[1], 4u);

  // Second load should have broadcast [8][4] shape
  const TensorShape &Load2D_2Shape = Ripple->getRippleShape(Load2D_2);
  EXPECT_EQ(Load2D_2Shape.rank(), 2u);
  EXPECT_EQ(Load2D_2Shape[0], 8u);
  EXPECT_EQ(Load2D_2Shape[1], 4u);

  // Subtraction should preserve [8][4] shape
  const TensorShape &Sub2DShape = Ripple->getRippleShape(Sub2D);
  EXPECT_EQ(Sub2DShape.rank(), 2u);
  EXPECT_EQ(Sub2DShape[0], 8u);
  EXPECT_EQ(Sub2DShape[1], 4u);

  // Final load should have [8][4] shape
  const TensorShape &Load2D_FinalShape = Ripple->getRippleShape(Load2D_Final);
  EXPECT_EQ(Load2D_FinalShape.rank(), 2u);
  EXPECT_EQ(Load2D_FinalShape[0], 8u);
  EXPECT_EQ(Load2D_FinalShape[1], 4u);
}

TEST_F(PropagateShapeTest, AllocaPromotion2DtoScalar) {
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

  // Create tensor shapes for testing
  Value *PE = IRB.getInt64(0);
  Value *One = IRB.getInt64(1);
  Value *Dim8 = IRB.getInt64(8);
  Value *Dim4 = IRB.getInt64(4);

  // Source shape: [8][4] (2D)
  AssertingVH<CallInst> SetShape2D = IRB.CreateCall(
      SetShapeFn, {PE, Dim8, Dim4, One, One, One, One, One, One, One, One});

  // Create tensors for testing
  Value *Idx0Val = IRB.getInt64(0);
  AssertingVH<CallInst> Index2D_Dim0 = IRB.CreateCall(
      IndexFn, {SetShape2D, Idx0Val}); // Creates [8] tensor from [8][4] shape

  Value *Idx1Val = IRB.getInt64(1);
  AssertingVH<CallInst> Index2D_Dim1 = IRB.CreateCall(
      IndexFn,
      {SetShape2D, Idx1Val}); // Creates [1][4] tensor from [8][4] shape

  // Create scalar values
  Value *Scalar5 = IRB.getInt64(5);
  Value *Scalar10 = IRB.getInt64(10);

  // Test: Alloca that progresses from 2D -> 1D -> scalar
  Type *Int64Ty = IRB.getInt64Ty();
  AssertingVH<Value> Alloca = IRB.CreateAlloca(Int64Ty, nullptr, "alloca");

  // Step 1: Start with 2D tensor [8][4] (via broadcast from [8] + [1][4])
  AssertingVH<Value> Initial2D =
      IRB.CreateAdd(Index2D_Dim0, Index2D_Dim1, "initial_2d");
  IRB.CreateStore(Initial2D, Alloca);

  // Load the 2D tensor from alloca
  AssertingVH<Value> Load_2D = IRB.CreateLoad(Int64Ty, Alloca, "load_2d");

  // Step 2: Reduce to 1D tensor [8] (use only dimension 0)
  AssertingVH<Value> Reduce_to_1D =
      IRB.CreateAdd(Index2D_Dim0, Scalar5, "reduce_to_1d");

  // Store the 1D result back to alloca
  IRB.CreateStore(Reduce_to_1D, Alloca);

  // Load the 1D tensor from alloca
  AssertingVH<Value> Load_1D = IRB.CreateLoad(Int64Ty, Alloca, "load_1d");

  // Step 3: Reduce to scalar (use scalar operations)
  AssertingVH<Value> Reduce_to_Scalar =
      IRB.CreateAdd(Scalar10, Scalar5, "reduce_to_scalar");

  // Store the scalar result back to alloca
  IRB.CreateStore(Reduce_to_Scalar, Alloca);

  // Final load to verify scalar
  AssertingVH<Value> Load_Scalar =
      IRB.CreateLoad(Int64Ty, Alloca, "load_scalar");

  // Perform final scalar operation
  AssertingVH<Value> Final_Scalar =
      IRB.CreateMul(Load_Scalar, Scalar5, "final_scalar");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify shape progression: 2D -> 1D -> scalar
  // Alloca should end up with [8][4]
  const TensorShape &AllocaShape = Ripple->getRippleShape(Alloca);
  EXPECT_EQ(AllocaShape.rank(), 2u);
  EXPECT_EQ(AllocaShape[0], 8u);
  EXPECT_EQ(AllocaShape[1], 4u);

  // Initial 2D operation should create [8][4] shape
  const TensorShape &Initial2DShape = Ripple->getRippleShape(Initial2D);
  EXPECT_EQ(Initial2DShape.rank(), 2u);
  EXPECT_EQ(Initial2DShape[0], 8u);
  EXPECT_EQ(Initial2DShape[1], 4u);

  // First load should have 2D shape
  const TensorShape &Load_2DShape = Ripple->getRippleShape(Load_2D);
  EXPECT_EQ(Load_2DShape.rank(), 2u);
  EXPECT_EQ(Load_2DShape[0], 8u);
  EXPECT_EQ(Load_2DShape[1], 4u);

  // Reduction to 1D should result in [8] shape
  const TensorShape &Reduce_to_1DShape = Ripple->getRippleShape(Reduce_to_1D);
  EXPECT_EQ(Reduce_to_1DShape.rank(), 2u);
  EXPECT_EQ(Reduce_to_1DShape[0], 8u);
  EXPECT_EQ(Reduce_to_1DShape[1], 1u);

  // Second load should have 1D shape
  const TensorShape &Load_1DShape = Ripple->getRippleShape(Load_1D);
  EXPECT_EQ(Load_1DShape.rank(), 2u);
  EXPECT_EQ(Load_1DShape[0], 8u);
  EXPECT_EQ(Load_1DShape[1], 1u);

  // Reduction to scalar should be scalar
  const TensorShape &Reduce_to_ScalarShape =
      Ripple->getRippleShape(Reduce_to_Scalar);
  EXPECT_TRUE(Reduce_to_ScalarShape.isScalar());

  // Final load should be scalar
  const TensorShape &Load_ScalarShape = Ripple->getRippleShape(Load_Scalar);
  EXPECT_TRUE(Load_ScalarShape.isScalar());

  // Final operation should remain scalar
  const TensorShape &Final_ScalarShape = Ripple->getRippleShape(Final_Scalar);
  EXPECT_TRUE(Final_ScalarShape.isScalar());
}

TEST_F(PropagateShapeTest, AllocaIncompatibleShapes) {
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

  // Create multiple incompatible tensor shapes for testing
  Value *PE = IRB.getInt64(0);
  Value *One = IRB.getInt64(1);

  // Shape 1: [3][7] (2D) - flatShape = 21
  Value *Dim3 = IRB.getInt64(3);
  Value *Dim7 = IRB.getInt64(7);
  AssertingVH<CallInst> SetShape1 = IRB.CreateCall(
      SetShapeFn, {PE, Dim3, Dim7, One, One, One, One, One, One, One, One});

  // Shape 2: [5][2] (2D) - flatShape = 10
  Value *Dim5 = IRB.getInt64(5);
  Value *Dim2 = IRB.getInt64(2);
  AssertingVH<CallInst> SetShape2 = IRB.CreateCall(
      SetShapeFn, {PE, Dim5, Dim2, One, One, One, One, One, One, One, One});

  // Shape 3: [4][6] (2D) - flatShape = 24
  Value *Dim4 = IRB.getInt64(4);
  Value *Dim6 = IRB.getInt64(6);
  AssertingVH<CallInst> SetShape3 = IRB.CreateCall(
      SetShapeFn, {PE, Dim4, Dim6, One, One, One, One, One, One, One, One});

  // Shape 4: [8] (1D) - flatShape = 8
  Value *Dim8 = IRB.getInt64(8);
  AssertingVH<CallInst> SetShape4 = IRB.CreateCall(
      SetShapeFn, {PE, Dim8, One, One, One, One, One, One, One, One, One});

  // Create tensors from these shapes
  Value *Idx0Val = IRB.getInt64(0);
  Value *Idx1Val = IRB.getInt64(1);

  // Tensor 1: [3] from [3][7] shape
  AssertingVH<CallInst> Tensor1 =
      IRB.CreateCall(IndexFn, {SetShape1, Idx0Val}); // [3]

  // Tensor 2: [1][7] from [3][7] shape
  AssertingVH<CallInst> Tensor2 =
      IRB.CreateCall(IndexFn, {SetShape1, Idx1Val}); // [1][7]

  // Tensor 3: [5] from [5][2] shape
  AssertingVH<CallInst> Tensor3 =
      IRB.CreateCall(IndexFn, {SetShape2, Idx0Val}); // [5]

  // Tensor 4: [1][2] from [5][2] shape
  AssertingVH<CallInst> Tensor4 =
      IRB.CreateCall(IndexFn, {SetShape2, Idx1Val}); // [1][2]

  // Tensor 5: [4] from [4][6] shape
  AssertingVH<CallInst> Tensor5 =
      IRB.CreateCall(IndexFn, {SetShape3, Idx0Val}); // [4]

  // Tensor 6: [1][6] from [4][6] shape
  AssertingVH<CallInst> Tensor6 =
      IRB.CreateCall(IndexFn, {SetShape3, Idx1Val}); // [1][6]

  // Tensor 7: [8] from [8] shape
  AssertingVH<CallInst> Tensor7 =
      IRB.CreateCall(IndexFn, {SetShape4, Idx0Val}); // [8]

  // Create scalar values
  Value *Scalar1 = IRB.getInt64(1);

  // Test: Alloca that stores multiple incompatible tensor shapes
  Type *Int64Ty = IRB.getInt64Ty();
  AssertingVH<Value> AllocaIncompat =
      IRB.CreateAlloca(Int64Ty, nullptr, "alloca_incompat");

  // Store Tensor1 [3] - flatShape = 3
  IRB.CreateStore(Tensor1, AllocaIncompat);
  AssertingVH<Value> Load1 = IRB.CreateLoad(Int64Ty, AllocaIncompat, "load1");

  // Create and store a broadcast result [3][7] - flatShape = 21
  AssertingVH<Value> Broadcast1 =
      IRB.CreateAdd(Tensor1, Tensor2, "broadcast1"); // [3] + [1][7] -> [3][7]
  IRB.CreateStore(Broadcast1, AllocaIncompat);
  AssertingVH<Value> Load2 = IRB.CreateLoad(Int64Ty, AllocaIncompat, "load2");

  // Store Tensor3 [5] - flatShape = 5 (incompatible with previous shapes)
  IRB.CreateStore(Tensor3, AllocaIncompat);
  AssertingVH<Value> Load3 = IRB.CreateLoad(Int64Ty, AllocaIncompat, "load3");

  // Create and store another broadcast result [5][2] - flatShape = 10
  AssertingVH<Value> Broadcast2 =
      IRB.CreateAdd(Tensor3, Tensor4, "broadcast2"); // [5] + [1][2] -> [5][2]
  IRB.CreateStore(Broadcast2, AllocaIncompat);
  AssertingVH<Value> Load4 = IRB.CreateLoad(Int64Ty, AllocaIncompat, "load4");

  // Store Tensor7 [8] - flatShape = 8 (incompatible with previous shapes)
  IRB.CreateStore(Tensor7, AllocaIncompat);
  AssertingVH<Value> Load5 = IRB.CreateLoad(Int64Ty, AllocaIncompat, "load5");

  // Create and store the largest broadcast result [4][6] - flatShape = 24
  AssertingVH<Value> Broadcast3 =
      IRB.CreateAdd(Tensor5, Tensor6, "broadcast3"); // [4] + [1][6] -> [4][6]
  IRB.CreateStore(Broadcast3, AllocaIncompat);
  AssertingVH<Value> Load6 = IRB.CreateLoad(Int64Ty, AllocaIncompat, "load6");

  // Store a scalar - flatShape = 1
  IRB.CreateStore(Scalar1, AllocaIncompat);
  AssertingVH<Value> Load7 = IRB.CreateLoad(Int64Ty, AllocaIncompat, "load7");

  IRB.CreateRetVoid();

  ASSERT_FALSE(verifyFunction(*F, &errs()));

  auto Ripple = createRipple(*F);
  ASSERT_TRUE(Ripple);
  bool WaitingForSpec = false;
  auto Err = Ripple->propagateShapes(WaitingForSpec);
  ASSERT_FALSE(Err);

  // Verify that the alloca shape is large enough to accommodate all stored
  // tensors The alloca should have the largest shape that can fit all stored
  // tensors Expected: [4][6] since it has the largest flatShape = 24
  const TensorShape &AllocaIncompatShape =
      Ripple->getRippleShape(AllocaIncompat);
  EXPECT_EQ(AllocaIncompatShape.rank(), 2u);
  EXPECT_EQ(AllocaIncompatShape[0], 4u);
  EXPECT_EQ(AllocaIncompatShape[1], 6u);

  // Verify that AllocaTensorShape.flatShape() >= all stored tensor flatShapes
  uint64_t allocaFlatShape = AllocaIncompatShape.flatShape();

  // Check individual tensor shapes and their flatShapes
  const TensorShape &Tensor1Shape =
      Ripple->getRippleShape(static_cast<Value *>(Tensor1));
  EXPECT_EQ(Tensor1Shape.rank(), 2u);
  EXPECT_EQ(Tensor1Shape[0], 3u);
  EXPECT_EQ(Tensor1Shape[1], 1u);
  EXPECT_GE(allocaFlatShape, Tensor1Shape.flatShape()); // 24 >= 3

  const TensorShape &Tensor2Shape =
      Ripple->getRippleShape(static_cast<Value *>(Tensor2));
  EXPECT_EQ(Tensor2Shape.rank(), 2u);
  EXPECT_EQ(Tensor2Shape[0], 1u);
  EXPECT_EQ(Tensor2Shape[1], 7u);
  EXPECT_GE(allocaFlatShape, Tensor2Shape.flatShape()); // 24 >= 7

  const TensorShape &Tensor3Shape =
      Ripple->getRippleShape(static_cast<Value *>(Tensor3));
  EXPECT_EQ(Tensor3Shape.rank(), 2u);
  EXPECT_EQ(Tensor3Shape[0], 5u);
  EXPECT_EQ(Tensor3Shape[1], 1u);
  EXPECT_GE(allocaFlatShape, Tensor3Shape.flatShape()); // 24 >= 5

  const TensorShape &Tensor4Shape =
      Ripple->getRippleShape(static_cast<Value *>(Tensor4));
  EXPECT_EQ(Tensor4Shape.rank(), 2u);
  EXPECT_EQ(Tensor4Shape[0], 1u);
  EXPECT_EQ(Tensor4Shape[1], 2u);
  EXPECT_GE(allocaFlatShape, Tensor4Shape.flatShape()); // 24 >= 2

  const TensorShape &Tensor5Shape =
      Ripple->getRippleShape(static_cast<Value *>(Tensor5));
  EXPECT_EQ(Tensor5Shape.rank(), 2u);
  EXPECT_EQ(Tensor5Shape[0], 4u);
  EXPECT_EQ(Tensor5Shape[1], 1u);
  EXPECT_GE(allocaFlatShape, Tensor5Shape.flatShape()); // 24 >= 4

  const TensorShape &Tensor6Shape =
      Ripple->getRippleShape(static_cast<Value *>(Tensor6));
  EXPECT_EQ(Tensor6Shape.rank(), 2u);
  EXPECT_EQ(Tensor6Shape[0], 1u);
  EXPECT_EQ(Tensor6Shape[1], 6u);
  EXPECT_GE(allocaFlatShape, Tensor6Shape.flatShape()); // 24 >= 6

  const TensorShape &Tensor7Shape =
      Ripple->getRippleShape(static_cast<Value *>(Tensor7));
  EXPECT_EQ(Tensor7Shape.rank(), 2u);
  EXPECT_EQ(Tensor7Shape[0], 8u);
  EXPECT_EQ(Tensor7Shape[1], 1u);
  EXPECT_GE(allocaFlatShape, Tensor7Shape.flatShape()); // 24 >= 8

  const TensorShape &Broadcast1Shape = Ripple->getRippleShape(Broadcast1);
  EXPECT_EQ(Broadcast1Shape.rank(), 2u);
  EXPECT_EQ(Broadcast1Shape[0], 3u);
  EXPECT_EQ(Broadcast1Shape[1], 7u);
  EXPECT_GE(allocaFlatShape, Broadcast1Shape.flatShape()); // 24 >= 21

  const TensorShape &Broadcast2Shape = Ripple->getRippleShape(Broadcast2);
  EXPECT_EQ(Broadcast2Shape.rank(), 2u);
  EXPECT_EQ(Broadcast2Shape[0], 5u);
  EXPECT_EQ(Broadcast2Shape[1], 2u);
  EXPECT_GE(allocaFlatShape, Broadcast2Shape.flatShape()); // 24 >= 10

  const TensorShape &Broadcast3Shape = Ripple->getRippleShape(Broadcast3);
  EXPECT_EQ(Broadcast3Shape.rank(), 2u);
  EXPECT_EQ(Broadcast3Shape[0], 4u);
  EXPECT_EQ(Broadcast3Shape[1], 6u);
  EXPECT_GE(allocaFlatShape, Broadcast3Shape.flatShape()); // 24 >= 24

  // Verify scalar storage (flatShape = 1)
  EXPECT_GE(allocaFlatShape, 1u); // 24 >= 1

  // Verify loads have appropriate shapes based on what was stored
  const TensorShape &Load1Shape = Ripple->getRippleShape(Load1);
  EXPECT_EQ(Load1Shape.rank(), 2u);
  EXPECT_EQ(Load1Shape[0], 3u);
  EXPECT_EQ(Load1Shape[1], 1u);

  const TensorShape &Load2Shape = Ripple->getRippleShape(Load2);
  EXPECT_EQ(Load2Shape.rank(), 2u);
  EXPECT_EQ(Load2Shape[0], 3u);
  EXPECT_EQ(Load2Shape[1], 7u);

  const TensorShape &Load3Shape = Ripple->getRippleShape(Load3);
  EXPECT_EQ(Load3Shape.rank(), 2u);
  EXPECT_EQ(Load3Shape[0], 5u);
  EXPECT_EQ(Load3Shape[1], 1u);

  const TensorShape &Load4Shape = Ripple->getRippleShape(Load4);
  EXPECT_EQ(Load4Shape.rank(), 2u);
  EXPECT_EQ(Load4Shape[0], 5u);
  EXPECT_EQ(Load4Shape[1], 2u);

  const TensorShape &Load5Shape = Ripple->getRippleShape(Load5);
  EXPECT_EQ(Load5Shape.rank(), 2u);
  EXPECT_EQ(Load5Shape[0], 8u);
  EXPECT_EQ(Load5Shape[1], 1u);

  const TensorShape &Load6Shape = Ripple->getRippleShape(Load6);
  EXPECT_EQ(Load6Shape.rank(), 2u);
  EXPECT_EQ(Load6Shape[0], 4u);
  EXPECT_EQ(Load6Shape[1], 6u);

  const TensorShape &Load7Shape = Ripple->getRippleShape(Load7);
  EXPECT_TRUE(Load7Shape.isScalar());
}

} // namespace
} // namespace llvm
