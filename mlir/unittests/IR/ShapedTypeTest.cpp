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
      MemRefType::Builder(memrefOriginalShape, memrefOriginalType)
          .setMemorySpace(memSpace)
          .setAffineMaps(map);
  // Update shape.
  llvm::SmallVector<int64_t> memrefNewShape({30, 40});
  ASSERT_NE(memrefOriginalShape, memrefNewShape);
  ASSERT_EQ(memrefType.clone(memrefNewShape),
            (MemRefType)MemRefType::Builder(memrefNewShape, memrefOriginalType)
                .setMemorySpace(memSpace)
                .setAffineMaps(map));
  // Update type.
  Type memrefNewType = f32;
  ASSERT_NE(memrefOriginalType, memrefNewType);
  ASSERT_EQ(memrefType.clone(memrefNewType),
            (MemRefType)MemRefType::Builder(memrefOriginalShape, memrefNewType)
                .setMemorySpace(memSpace)
                .setAffineMaps(map));
  // Update both.
  ASSERT_EQ(memrefType.clone(memrefNewShape, memrefNewType),
            (MemRefType)MemRefType::Builder(memrefNewShape, memrefNewType)
                .setMemorySpace(memSpace)
                .setAffineMaps(map));

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
  ASSERT_EQ(tensorType.clone(tensorNewShape),
            RankedTensorType::get(tensorNewShape, tensorOriginalType));
  // Update type.
  Type tensorNewType = f32;
  ASSERT_NE(tensorOriginalType, tensorNewType);
  ASSERT_EQ(tensorType.clone(tensorNewType),
            RankedTensorType::get(tensorOriginalShape, tensorNewType));
  // Update both.
  ASSERT_EQ(tensorType.clone(tensorNewShape, tensorNewType),
            RankedTensorType::get(tensorNewShape, tensorNewType));

  // Test unranked tensor cloning.
  ShapedType unrankedTensorType = UnrankedTensorType::get(tensorOriginalType);
  ASSERT_EQ(unrankedTensorType.clone(tensorNewShape),
            RankedTensorType::get(tensorNewShape, tensorOriginalType));
  ASSERT_EQ(unrankedTensorType.clone(tensorNewType),
            UnrankedTensorType::get(tensorNewType));
  ASSERT_EQ(unrankedTensorType.clone(tensorNewShape),
            RankedTensorType::get(tensorNewShape, tensorOriginalType));
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

} // end namespace
