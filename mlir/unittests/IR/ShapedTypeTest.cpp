//===- ShapedTypeTest.cpp - ShapedType unit tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::detail;

namespace {
TEST(ShapedTypeTest, CloneMemref) {
  MLIRContext context;

  Type i32 = IntegerType::get(&context, 32);
  Type f32 = FloatType::getF32(&context);
  Attribute memSpace = IntegerAttr::get(IntegerType::get(&context, 64), 7);
  Type memrefOriginalType = i32;
  llvm::SmallVector<int64_t> memrefOriginalShape({10, 20});
  AffineMap map = makeStridedLinearLayoutMap({2, 3}, 5, &context);

  ShapedType memrefType =
      (ShapedType)MemRefType::Builder(memrefOriginalShape, memrefOriginalType)
          .setMemorySpace(memSpace)
          .setLayout(AffineMapAttr::get(map));
  // Update shape.
  llvm::SmallVector<int64_t> memrefNewShape({30, 40});
  ASSERT_NE(memrefOriginalShape, memrefNewShape);
  ASSERT_EQ(memrefType.clone(memrefNewShape),
            (ShapedType)MemRefType::Builder(memrefNewShape, memrefOriginalType)
                .setMemorySpace(memSpace)
                .setLayout(AffineMapAttr::get(map)));
  // Update type.
  Type memrefNewType = f32;
  ASSERT_NE(memrefOriginalType, memrefNewType);
  ASSERT_EQ(memrefType.clone(memrefNewType),
            (MemRefType)MemRefType::Builder(memrefOriginalShape, memrefNewType)
                .setMemorySpace(memSpace)
                .setLayout(AffineMapAttr::get(map)));
  // Update both.
  ASSERT_EQ(memrefType.clone(memrefNewShape, memrefNewType),
            (MemRefType)MemRefType::Builder(memrefNewShape, memrefNewType)
                .setMemorySpace(memSpace)
                .setLayout(AffineMapAttr::get(map)));

  // Test unranked memref cloning.
  ShapedType unrankedTensorType =
      UnrankedMemRefType::get(memrefOriginalType, memSpace);
  ASSERT_EQ(unrankedTensorType.clone(memrefNewShape),
            (MemRefType)MemRefType::Builder(memrefNewShape, memrefOriginalType)
                .setMemorySpace(memSpace));
  ASSERT_EQ(unrankedTensorType.clone(memrefNewType),
            UnrankedMemRefType::get(memrefNewType, memSpace));
  ASSERT_EQ(unrankedTensorType.clone(memrefNewShape, memrefNewType),
            (MemRefType)MemRefType::Builder(memrefNewShape, memrefNewType)
                .setMemorySpace(memSpace));
}

TEST(ShapedTypeTest, CloneTensor) {
  MLIRContext context;

  Type i32 = IntegerType::get(&context, 32);
  Type f32 = FloatType::getF32(&context);

  Type tensorOriginalType = i32;
  llvm::SmallVector<int64_t> tensorOriginalShape({10, 20});

  // Test ranked tensor cloning.
  ShapedType tensorType =
      RankedTensorType::get(tensorOriginalShape, tensorOriginalType);
  // Update shape.
  llvm::SmallVector<int64_t> tensorNewShape({30, 40});
  ASSERT_NE(tensorOriginalShape, tensorNewShape);
  ASSERT_EQ(
      tensorType.clone(tensorNewShape),
      (ShapedType)RankedTensorType::get(tensorNewShape, tensorOriginalType));
  // Update type.
  Type tensorNewType = f32;
  ASSERT_NE(tensorOriginalType, tensorNewType);
  ASSERT_EQ(
      tensorType.clone(tensorNewType),
      (ShapedType)RankedTensorType::get(tensorOriginalShape, tensorNewType));
  // Update both.
  ASSERT_EQ(tensorType.clone(tensorNewShape, tensorNewType),
            (ShapedType)RankedTensorType::get(tensorNewShape, tensorNewType));

  // Test unranked tensor cloning.
  ShapedType unrankedTensorType = UnrankedTensorType::get(tensorOriginalType);
  ASSERT_EQ(
      unrankedTensorType.clone(tensorNewShape),
      (ShapedType)RankedTensorType::get(tensorNewShape, tensorOriginalType));
  ASSERT_EQ(unrankedTensorType.clone(tensorNewType),
            (ShapedType)UnrankedTensorType::get(tensorNewType));
  ASSERT_EQ(
      unrankedTensorType.clone(tensorNewShape),
      (ShapedType)RankedTensorType::get(tensorNewShape, tensorOriginalType));
}

TEST(ShapedTypeTest, CloneVector) {
  MLIRContext context;

  Type i32 = IntegerType::get(&context, 32);
  Type f32 = FloatType::getF32(&context);

  Type vectorOriginalType = i32;
  llvm::SmallVector<int64_t> vectorOriginalShape({10, 20});
  ShapedType vectorType =
      VectorType::get(vectorOriginalShape, vectorOriginalType);
  // Update shape.
  llvm::SmallVector<int64_t> vectorNewShape({30, 40});
  ASSERT_NE(vectorOriginalShape, vectorNewShape);
  ASSERT_EQ(vectorType.clone(vectorNewShape),
            VectorType::get(vectorNewShape, vectorOriginalType));
  // Update type.
  Type vectorNewType = f32;
  ASSERT_NE(vectorOriginalType, vectorNewType);
  ASSERT_EQ(vectorType.clone(vectorNewType),
            VectorType::get(vectorOriginalShape, vectorNewType));
  // Update both.
  ASSERT_EQ(vectorType.clone(vectorNewShape, vectorNewType),
            VectorType::get(vectorNewShape, vectorNewType));
}

TEST(ShapedTypeTest, VectorTypeBuilder) {
  MLIRContext context;
  Type f32 = FloatType::getF32(&context);

  VectorType vectorType =
      VectorType::get({2, 4, 8, 9, 1}, f32, {true, false, true, false, false});

  {
    // Drop some dims.
    VectorType dropFrontTwoDims =
        VectorType::Builder(vectorType).dropDim(0).dropDim(0);
    ASSERT_EQ(vectorType.getElementType(), dropFrontTwoDims.getElementType());
    ASSERT_EQ(vectorType.getShape().drop_front(2), dropFrontTwoDims.getShape());
    ASSERT_EQ(vectorType.getScalableDims().drop_front(2),
              dropFrontTwoDims.getScalableDims());
  }

  {
    // Set some dims.
    VectorType setTwoDims =
        VectorType::Builder(vectorType).setDim(0, 10).setDim(3, 12);
    ASSERT_EQ(setTwoDims.getShape(), ArrayRef<int64_t>({10, 4, 8, 12, 1}));
    ASSERT_EQ(vectorType.getElementType(), setTwoDims.getElementType());
    ASSERT_EQ(vectorType.getScalableDims(), setTwoDims.getScalableDims());
  }

  {
    // Test for bug from:
    // https://github.com/llvm/llvm-project/commit/b44b3494f60296db6aca38a14cab061d9b747a0a
    // Constructs a temporary builder, modifies it, copies it to `builder`.
    VectorType::Builder builder = VectorType::Builder(vectorType).setDim(0, 16);
    VectorType newVectorType = VectorType(builder);
    ASSERT_EQ(newVectorType.getDimSize(0), 16);
  }
}

TEST(ShapedTypeTest, RankedTensorTypeBuilder) {
  MLIRContext context;
  Type f32 = FloatType::getF32(&context);

  RankedTensorType tensorType = RankedTensorType::get({2, 4, 8, 16, 32}, f32);

  {
    // Drop some dims.
    RankedTensorType dropFrontTwoDims =
        RankedTensorType::Builder(tensorType).dropDim(0).dropDim(0);
    ASSERT_EQ(tensorType.getElementType(), dropFrontTwoDims.getElementType());
    ASSERT_EQ(tensorType.getShape().drop_front(2), dropFrontTwoDims.getShape());
  }

  {
    // Insert some dims.
    RankedTensorType insertTwoDims =
        RankedTensorType::Builder(tensorType).insertDim(7, 2).insertDim(9, 3);
    ASSERT_EQ(tensorType.getElementType(), insertTwoDims.getElementType());
    ASSERT_EQ(insertTwoDims.getShape(),
              ArrayRef<int64_t>({2, 4, 7, 9, 8, 16, 32}));
  }

  {
    // Test for bug from:
    // https://github.com/llvm/llvm-project/commit/b44b3494f60296db6aca38a14cab061d9b747a0a
    // Constructs a temporary builder, modifies it, copies it to `builder`.
    RankedTensorType::Builder builder =
        RankedTensorType::Builder(tensorType).dropDim(0);
    RankedTensorType newTensorType = RankedTensorType(builder);
    ASSERT_EQ(tensorType.getShape().drop_front(), newTensorType.getShape());
  }
}

} // namespace
