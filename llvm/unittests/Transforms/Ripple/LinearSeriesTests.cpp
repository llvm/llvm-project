//===- LinearSeriesTests.cpp - -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RippleTestBase.h"
#include "llvm/Transforms/Ripple/Ripple.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

using LinearSeriesTest = RippleTestBase;

// Test 1: Scalar Linear Series (both base and slope are scalar)
TEST_F(LinearSeriesTest, ScalarLinearSeries) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(42);
  TensorShape BaseShape; // Scalar shape (rank 0)
  SmallVector<Value *> Slopes = {};
  TensorShape SlopeShape; // Scalar shape (rank 0)

  LinearSeries LS(Base, BaseShape, Slopes, SlopeShape);

  // Verify it's scalar
  EXPECT_TRUE(LS.isScalar());
  EXPECT_FALSE(LS.hasSlope());
  EXPECT_EQ(LS.rank(), 0u);
  EXPECT_EQ(LS.getBase(), Base);
  EXPECT_EQ(LS.getBaseShape(), BaseShape);
  EXPECT_EQ(LS.getSlopeShape(), SlopeShape);
  EXPECT_EQ(LS.getShape(), BaseShape);
}

// Test 2: 1D Linear Series with scalar base and vector slope
TEST_F(LinearSeriesTest, OneDimensionalLinearSeries) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(10);
  TensorShape BaseShape(1, 0, 1); // Scalar in dimension 0: [1]
  SmallVector<Value *> Slopes = {IRB.getInt32(1)};
  TensorShape SlopeShape(1, 0, 32); // Vector in dimension 0: [32]

  LinearSeries LS(Base, BaseShape, Slopes, SlopeShape);

  // Verify properties
  EXPECT_FALSE(LS.isScalar());
  EXPECT_TRUE(LS.hasSlope());
  EXPECT_EQ(LS.rank(), 1u);
  EXPECT_EQ(LS.getBase(), Base);
  EXPECT_EQ(LS.getSlope(0), Slopes[0]);
  EXPECT_EQ(LS.getShape(0), 32u);

  // Verify base and slope shapes are orthogonal
  EXPECT_EQ(LS.getBaseShape()[0], 1u);
  EXPECT_EQ(LS.getSlopeShape()[0], 32u);

  // Verify combined shape
  TensorShape ExpectedShape(1, 0, 32);
  EXPECT_EQ(LS.getShape(), ExpectedShape);
}

// Test 3: Tensor base with scalar slope (no effective slope)
TEST_F(LinearSeriesTest, TensorBaseScalarSlope) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(100);
  TensorShape BaseShape(1, 0, 16); // Vector base: [16]
  SmallVector<Value *> Slopes = {IRB.getInt32(0)};
  TensorShape SlopeShape(1, 0, 1); // Scalar slope: [1]

  LinearSeries LS(Base, BaseShape, Slopes, SlopeShape);

  // Verify properties
  EXPECT_FALSE(LS.isScalar());
  EXPECT_FALSE(LS.hasSlope()); // hasSlope checks if slope shape is non-scalar
  EXPECT_EQ(LS.rank(), 1u);
  EXPECT_EQ(LS.getBase(), Base);
  EXPECT_EQ(LS.getShape(0), 16u);

  // Verify shapes
  EXPECT_EQ(LS.getBaseShape()[0], 16u);
  EXPECT_EQ(LS.getSlopeShape()[0], 1u);

  // Verify combined shape
  TensorShape ExpectedShape(1, 0, 16);
  EXPECT_EQ(LS.getShape(), ExpectedShape);
}

// Test 4: Orthogonal shapes - base and slope on different dimensions
TEST_F(LinearSeriesTest, OrthogonalShapes) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(5);
  TensorShape BaseShape(TensorShape::Shape({32, 1})); // [32, 1]
  SmallVector<Value *> Slopes = {IRB.getInt32(0), IRB.getInt32(1)};
  TensorShape SlopeShape(TensorShape::Shape({1, 16})); // [1, 16]

  LinearSeries LS(Base, BaseShape, Slopes, SlopeShape);

  // Verify properties
  EXPECT_FALSE(LS.isScalar());
  EXPECT_TRUE(LS.hasSlope());
  EXPECT_EQ(LS.rank(), 2u);
  EXPECT_EQ(LS.getBase(), Base);

  // Verify individual dimension shapes
  EXPECT_EQ(LS.getShape(0), 32u);
  EXPECT_EQ(LS.getShape(1), 16u);

  // Verify base and slope are orthogonal
  EXPECT_EQ(LS.getBaseShape()[0], 32u);
  EXPECT_EQ(LS.getBaseShape()[1], 1u);
  EXPECT_EQ(LS.getSlopeShape()[0], 1u);
  EXPECT_EQ(LS.getSlopeShape()[1], 16u);

  // Verify combined shape
  TensorShape ExpectedShape(TensorShape::Shape({32, 16}));
  EXPECT_EQ(LS.getShape(), ExpectedShape);
}

// Test 5: Splat dimensions detection (zero slopes on non-trivial dimensions)
TEST_F(LinearSeriesTest, SplatDimensions) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(7);
  TensorShape BaseShape(TensorShape::Shape({1, 1, 1}));
  SmallVector<Value *> Slopes = {
      IRB.getInt32(1), // Non-zero slope
      IRB.getInt32(0), // Zero slope (splat)
      IRB.getInt32(0)  // Zero slope (splat)
  };
  TensorShape SlopeShape(TensorShape::Shape({8, 4, 2}));

  LinearSeries LS(Base, BaseShape, Slopes, SlopeShape);

  // Get splat dimensions
  BitVector SplatDims = LS.getSplatDims();

  // Verify splat detection
  EXPECT_EQ(SplatDims.size(), 3u);
  EXPECT_FALSE(SplatDims[0]); // Dimension 0 has non-zero slope
  EXPECT_TRUE(SplatDims[1]);  // Dimension 1 has zero slope and size > 1
  EXPECT_TRUE(SplatDims[2]);  // Dimension 2 has zero slope and size > 1
}

// Test 6: Splat dimensions with all non-zero slopes
TEST_F(LinearSeriesTest, NoSplatDimensions) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(3);
  TensorShape BaseShape(TensorShape::Shape({1, 1}));
  SmallVector<Value *> Slopes = {IRB.getInt32(1), IRB.getInt32(2)};
  TensorShape SlopeShape(TensorShape::Shape({8, 4}));

  LinearSeries LS(Base, BaseShape, Slopes, SlopeShape);

  // Get splat dimensions
  BitVector SplatDims = LS.getSplatDims();

  // Verify no splat dimensions
  EXPECT_EQ(SplatDims.size(), 2u);
  EXPECT_FALSE(SplatDims[0]);
  EXPECT_FALSE(SplatDims[1]);
}

// Test 7: Remove slopes from specific dimensions
TEST_F(LinearSeriesTest, RemoveSlopes) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(20);
  TensorShape BaseShape(TensorShape::Shape({1, 1, 1}));
  SmallVector<Value *> Slopes = {IRB.getInt32(1), IRB.getInt32(2),
                                 IRB.getInt32(3)};
  TensorShape SlopeShape(TensorShape::Shape({8, 4, 2}));

  LinearSeries LS(Base, BaseShape, Slopes, SlopeShape);

  // Remove slopes from dimensions 0 and 2
  BitVector DimsToRemove(3);
  DimsToRemove.set(0);
  DimsToRemove.set(2);

  LinearSeries NewLS = LS.removeSlopes(DimsToRemove);

  // Verify the new slope shape
  EXPECT_EQ(NewLS.getSlopeShape()[0], 1u); // Dimension 0 removed
  EXPECT_EQ(NewLS.getSlopeShape()[1], 4u); // Dimension 1 unchanged
  EXPECT_EQ(NewLS.getSlopeShape()[2], 1u); // Dimension 2 removed

  // Verify base and slopes are shared
  EXPECT_EQ(NewLS.getBase(), Base);
  EXPECT_EQ(NewLS.getSlope(0), Slopes[0]);
  EXPECT_EQ(NewLS.getSlope(1), Slopes[1]);
  EXPECT_EQ(NewLS.getSlope(2), Slopes[2]);
}

// Test 8: hasZeroSlopes method
TEST_F(LinearSeriesTest, HasZeroSlopes) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(15);

  // Case 1: All zero slopes
  TensorShape BaseShape1(TensorShape::Shape({1, 1}));
  SmallVector<Value *> ZeroSlopes = {IRB.getInt32(0), IRB.getInt32(0)};
  TensorShape SlopeShape1(TensorShape::Shape({8, 4}));
  LinearSeries LS1(Base, BaseShape1, ZeroSlopes, SlopeShape1);
  EXPECT_TRUE(LS1.hasZeroSlopes());

  // Case 2: Mixed slopes
  SmallVector<Value *> MixedSlopes = {IRB.getInt32(1), IRB.getInt32(0)};
  LinearSeries LS2(Base, BaseShape1, MixedSlopes, SlopeShape1);
  EXPECT_FALSE(LS2.hasZeroSlopes());

  // Case 3: Scalar slope shape
  TensorShape ScalarSlopeShape;
  SmallVector<Value *> EmptySlopes = {};
  LinearSeries LS3(Base, TensorShape(), EmptySlopes, ScalarSlopeShape);
  EXPECT_TRUE(LS3.hasZeroSlopes());
}

// Test 9: isScalarOrSplat method
TEST_F(LinearSeriesTest, IsScalarOrSplat) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(25);

  // Case 1: Scalar
  LinearSeries LS1(Base, TensorShape(), SmallVector<Value *>(), TensorShape());
  EXPECT_TRUE(LS1.isScalarOrSplat());

  // Case 2: Splat (zero slopes)
  TensorShape BaseShape2(TensorShape::Shape({1, 1}));
  SmallVector<Value *> ZeroSlopes = {IRB.getInt32(0), IRB.getInt32(0)};
  TensorShape SlopeShape2(TensorShape::Shape({8, 4}));
  LinearSeries LS2(Base, BaseShape2, ZeroSlopes, SlopeShape2);
  EXPECT_TRUE(LS2.isScalarOrSplat());

  // Case 3: Not a splat (has non-zero slopes)
  SmallVector<Value *> NonZeroSlopes = {IRB.getInt32(1), IRB.getInt32(0)};
  LinearSeries LS3(Base, BaseShape2, NonZeroSlopes, SlopeShape2);
  EXPECT_FALSE(LS3.isScalarOrSplat());
}

// Test 10: getSlopeTypeFor with integer base
TEST_F(LinearSeriesTest, GetSlopeTypeForInteger) {
  Module M("test", C);
  const DataLayout &DL = M.getDataLayout();

  Type *Int32Ty = Type::getInt32Ty(C);
  IntegerType *SlopeType = LinearSeries::getSlopeTypeFor(DL, Int32Ty);

  EXPECT_NE(SlopeType, nullptr);
  EXPECT_TRUE(SlopeType->isIntegerTy());
  EXPECT_EQ(SlopeType, Int32Ty);
}

// Test 11: getSlopeTypeFor with pointer base
TEST_F(LinearSeriesTest, GetSlopeTypeForPointer) {
  Module M("test", C);
  const DataLayout &DL = M.getDataLayout();

  Type *PtrTy = PointerType::get(C, 0);
  IntegerType *SlopeType = LinearSeries::getSlopeTypeFor(DL, PtrTy);

  EXPECT_NE(SlopeType, nullptr);
  EXPECT_TRUE(SlopeType->isIntegerTy());
  // The slope type should match the pointer size from the data layout
  EXPECT_EQ(SlopeType->getIntegerBitWidth(),
            DL.getPointerSizeInBits(PtrTy->getPointerAddressSpace()));
}

// Test 12: constructLinearSeriesVector
TEST_F(LinearSeriesTest, ConstructLinearSeriesVector) {
  IntegerType *Int32Ty = Type::getInt32Ty(C);
  uint64_t Size = 8;

  Constant *Series = LinearSeries::constructLinearSeriesVector(Int32Ty, Size);

  EXPECT_NE(Series, nullptr);
  EXPECT_TRUE(isa<ConstantDataVector>(Series));

  ConstantDataVector *CDV = cast<ConstantDataVector>(Series);
  EXPECT_EQ(CDV->getNumElements(), Size);

  // Verify the values are [0, 1, 2, 3, 4, 5, 6, 7]
  for (uint64_t i = 0; i < Size; ++i) {
    Constant *Elem = CDV->getElementAsConstant(i);
    ConstantInt *CI = dyn_cast<ConstantInt>(Elem);
    EXPECT_NE(CI, nullptr);
    EXPECT_EQ(CI->getZExtValue(), i);
  }
}

// Test 13: Multi-dimensional with pointer base
TEST_F(LinearSeriesTest, PointerBaseLinearSeries) {
  Module M("test", C);
  const DataLayout &DL = M.getDataLayout();
  IRBuilder<> IRB(C);

  Type *PtrTy = PointerType::get(C, 0);
  Value *PtrBase = Constant::getNullValue(PtrTy);

  TensorShape BaseShape(1, 0, 1); // [1]
  IntegerType *SlopeType = LinearSeries::getSlopeTypeFor(DL, PtrTy);
  SmallVector<Value *> Slopes = {ConstantInt::get(SlopeType, 4)};
  TensorShape SlopeShape(1, 0, 16); // [16]

  LinearSeries LS(PtrBase, BaseShape, Slopes, SlopeShape);

  EXPECT_FALSE(LS.isScalar());
  EXPECT_TRUE(LS.hasSlope());
  EXPECT_EQ(LS.rank(), 1u);
  EXPECT_EQ(LS.getBase(), PtrBase);
  EXPECT_EQ(LS.getShape(0), 16u);
}

// Test 14: 3D orthogonal shapes
TEST_F(LinearSeriesTest, ThreeDimensionalOrthogonal) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(0);
  TensorShape BaseShape(TensorShape::Shape({8, 1, 1})); // [8, 1, 1]
  SmallVector<Value *> Slopes = {IRB.getInt32(0), IRB.getInt32(1),
                                 IRB.getInt32(2)};
  TensorShape SlopeShape(TensorShape::Shape({1, 4, 2})); // [1, 4, 2]

  LinearSeries LS(Base, BaseShape, Slopes, SlopeShape);

  EXPECT_EQ(LS.rank(), 3u);
  EXPECT_EQ(LS.getShape(0), 8u);
  EXPECT_EQ(LS.getShape(1), 4u);
  EXPECT_EQ(LS.getShape(2), 2u);

  // Verify orthogonality
  EXPECT_EQ(LS.getBaseShape()[0], 8u);
  EXPECT_EQ(LS.getBaseShape()[1], 1u);
  EXPECT_EQ(LS.getBaseShape()[2], 1u);
  EXPECT_EQ(LS.getSlopeShape()[0], 1u);
  EXPECT_EQ(LS.getSlopeShape()[1], 4u);
  EXPECT_EQ(LS.getSlopeShape()[2], 2u);

  TensorShape ExpectedShape(TensorShape::Shape({8, 4, 2}));
  EXPECT_EQ(LS.getShape(), ExpectedShape);
}

// Test 15: Edge case - single element in each dimension
TEST_F(LinearSeriesTest, SingleElementDimensions) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(99);
  TensorShape BaseShape(TensorShape::Shape({1, 1, 1}));
  SmallVector<Value *> Slopes = {IRB.getInt32(0), IRB.getInt32(0),
                                 IRB.getInt32(0)};
  TensorShape SlopeShape(TensorShape::Shape({1, 1, 1}));

  LinearSeries LS(Base, BaseShape, Slopes, SlopeShape);

  // This should behave like a scalar
  EXPECT_TRUE(LS.isScalar());
  EXPECT_EQ(LS.rank(), 3u);
  EXPECT_EQ(LS.getShape(0), 1u);
  EXPECT_EQ(LS.getShape(1), 1u);
  EXPECT_EQ(LS.getShape(2), 1u);
}

#if GTEST_HAS_DEATH_TEST

// Test 16: Death test - non-orthogonal base and slope shapes
TEST_F(LinearSeriesTest, OrthogonalShapesDeathTest) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(42);

  // Both base and slope have non-trivial size in the same dimension
  // This violates the orthogonality constraint
  TensorShape BaseShape(1, 0, 32); // [32]
  SmallVector<Value *> Slopes = {IRB.getInt32(1)};
  TensorShape SlopeShape(1, 0, 32); // [32]

  // This should trigger llvm_unreachable in computeLSShape
  EXPECT_DEATH(
      { LinearSeries LS(Base, BaseShape, Slopes, SlopeShape); },
      "Base and slope shapes are not orthogonal");
}

// Test 17: Death test - non-orthogonal base and slope shapes (different sizes)
TEST_F(LinearSeriesTest, NonOrthogonalShapesDifferentSizesDeathTest) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(42);

  // Both base and slope have non-trivial size in the same dimension
  // This violates the orthogonality constraint
  TensorShape BaseShape(1, 0, 32); // [32]
  SmallVector<Value *> Slopes = {IRB.getInt32(1)};
  TensorShape SlopeShape(1, 0, 16); // [16]

  // This should trigger llvm_unreachable in computeLSShape
  EXPECT_DEATH(
      { LinearSeries LS(Base, BaseShape, Slopes, SlopeShape); },
      "Base and slope shapes are not orthogonal");
}

// Test 18: Death test - 2D non-orthogonal shapes
TEST_F(LinearSeriesTest, NonOrthogonalShapes2DDeathTest) {
  IRBuilder<> IRB(C);
  Value *Base = IRB.getInt32(10);

  // Both base and slope have non-trivial size in dimension 1
  TensorShape BaseShape(TensorShape::Shape({1, 8})); // [1, 8]
  SmallVector<Value *> Slopes = {IRB.getInt32(0), IRB.getInt32(1)};
  TensorShape SlopeShape(TensorShape::Shape({4, 2})); // [4, 2]

  // This should trigger llvm_unreachable
  EXPECT_DEATH(
      { LinearSeries LS(Base, BaseShape, Slopes, SlopeShape); },
      "Base and slope shapes are not orthogonal");
}

// Test 19: Death test - invalid base type (scalar float)
TEST_F(LinearSeriesTest, InvalidFloatBaseTypeDeathTest) {
  IRBuilder<> IRB(C);

  // Create a float base (invalid type for LinearSeries)
  Value *FloatBase = ConstantFP::get(Type::getFloatTy(C), 3.14);
  TensorShape BaseShape(1, 0, 1); // [1]
  SmallVector<Value *> Slopes = {IRB.getInt32(1)};
  TensorShape SlopeShape(1, 0, 32); // [32]

  // This should trigger llvm_unreachable in computeLSShape
  EXPECT_DEATH(
      { LinearSeries LS(FloatBase, BaseShape, Slopes, SlopeShape); },
      "LinearSeries base must be of integer or pointer type");
}

// Test 20: Death test - invalid base type (vector of floats)
TEST_F(LinearSeriesTest, InvalidFloatVectorBaseTypeDeathTest) {
  IRBuilder<> IRB(C);

  // Create a vector of 4 floats as base (invalid type for LinearSeries)
  Type *FloatTy = Type::getFloatTy(C);
  Value *FloatVectorBase = ConstantVector::getSplat(
      ElementCount::getFixed(4), ConstantFP::get(FloatTy, 1.5));

  TensorShape BaseShape(1, 0, 4); // [4] - tensor base
  SmallVector<Value *> Slopes = {IRB.getInt32(0)};
  TensorShape SlopeShape(1, 0, 1); // [1] - scalar slope

  // This should trigger llvm_unreachable in computeLSShape
  // The scalar type of the vector is float, which is invalid
  EXPECT_DEATH(
      { LinearSeries LS(FloatVectorBase, BaseShape, Slopes, SlopeShape); },
      "LinearSeries base must be of integer or pointer type");
}

#endif // GTEST_HAS_DEATH_TEST

} // namespace
} // namespace llvm
