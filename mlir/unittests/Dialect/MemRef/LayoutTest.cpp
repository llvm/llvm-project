//===- LayoutTest.cpp - unit tests related to memref layout --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::memref;

TEST(MemRefLayout, maxContigDim) {
  MLIRContext ctx;
  OpBuilder b(&ctx);

  const auto _ = ShapedType::kDynamic;
  const auto f32 = b.getF32Type();
  auto strided = [&ctx](ArrayRef<int64_t> s) {
    return StridedLayoutAttr::get(&ctx, 0, s);
  };

  // memref<2x2x2xf32, strided<[4,2,1]>
  auto m1 = MemRefType::get({2, 2, 2}, f32, strided({4, 2, 1}));
  EXPECT_EQ(m1.getNumContiguousTrailingDims(), 3);

  // memref<2x2x2xf32, strided<[8,2,1]>
  auto m2 = MemRefType::get({2, 2, 2}, f32, strided({8, 2, 1}));
  EXPECT_EQ(m2.getNumContiguousTrailingDims(), 2);

  // memref<2x2x2xf32, strided<[8,4,1]>
  auto m3 = MemRefType::get({2, 2, 2}, f32, strided({8, 4, 1}));
  EXPECT_EQ(m3.getNumContiguousTrailingDims(), 1);

  // memref<2x2x2xf32, strided<[8,4,2]>
  auto m4 = MemRefType::get({2, 2, 2}, f32, strided({8, 4, 2}));
  EXPECT_EQ(m4.getNumContiguousTrailingDims(), 0);

  // memref<2x2x?xf32, strided<[?,?,1]>
  auto m5 = MemRefType::get({2, 2, _}, f32, strided({_, _, 1}));
  EXPECT_EQ(m5.getNumContiguousTrailingDims(), 1);

  // memref<2x2x?xf32, strided<[?,?,2]>
  auto m6 = MemRefType::get({2, 2, _}, f32, strided({_, _, 2}));
  EXPECT_EQ(m6.getNumContiguousTrailingDims(), 0);

  // memref<2x?x2xf32, strided<[?,2,1]>
  auto m7 = MemRefType::get({2, _, 2}, f32, strided({_, 2, 1}));
  EXPECT_EQ(m7.getNumContiguousTrailingDims(), 2);

  // memref<2x?x2xf32, strided<[?,4,1]>
  auto m8 = MemRefType::get({2, _, 2}, f32, strided({_, 4, 1}));
  EXPECT_EQ(m8.getNumContiguousTrailingDims(), 1);

  // memref<2x?x2xf32, strided<[?,4,2]>
  auto m9 = MemRefType::get({2, _, 2}, f32, strided({_, 4, 2}));
  EXPECT_EQ(m9.getNumContiguousTrailingDims(), 0);

  // memref<?x2x2xf32, strided<[4,2,1]>
  auto m10 = MemRefType::get({_, 2, 2}, f32, strided({4, 2, 1}));
  EXPECT_EQ(m10.getNumContiguousTrailingDims(), 3);

  // memref<?x2x2xf32, strided<[8,2,1]>
  auto m11 = MemRefType::get({_, 2, 2}, f32, strided({8, 2, 1}));
  EXPECT_EQ(m11.getNumContiguousTrailingDims(), 2);

  // memref<?x2x2xf32, strided<[8,4,1]>
  auto m12 = MemRefType::get({_, 2, 2}, f32, strided({8, 4, 1}));
  EXPECT_EQ(m12.getNumContiguousTrailingDims(), 1);

  // memref<?x2x2xf32, strided<[8,4,2]>
  auto m13 = MemRefType::get({_, 2, 2}, f32, strided({8, 4, 2}));
  EXPECT_EQ(m13.getNumContiguousTrailingDims(), 0);

  // memref<2x2x1xf32, strided<[2,1,2]>
  auto m14 = MemRefType::get({2, 2, 1}, f32, strided({2, 1, 2}));
  EXPECT_EQ(m14.getNumContiguousTrailingDims(), 3);

  // memref<2x2x1xf32, strided<[2,1,?]>
  auto m15 = MemRefType::get({2, 2, 1}, f32, strided({2, 1, _}));
  EXPECT_EQ(m15.getNumContiguousTrailingDims(), 3);

  // memref<2x2x1xf32, strided<[4,2,2]>
  auto m16 = MemRefType::get({2, 2, 1}, f32, strided({4, 2, 2}));
  EXPECT_EQ(m16.getNumContiguousTrailingDims(), 1);

  // memref<2x1x2xf32, strided<[2,4,1]>
  auto m17 = MemRefType::get({2, 1, 2}, f32, strided({2, 4, 1}));
  EXPECT_EQ(m17.getNumContiguousTrailingDims(), 3);

  // memref<2x1x2xf32, strided<[2,?,1]>
  auto m18 = MemRefType::get({2, 1, 2}, f32, strided({2, _, 1}));
  EXPECT_EQ(m18.getNumContiguousTrailingDims(), 3);
}

TEST(MemRefLayout, contigTrailingDim) {
  MLIRContext ctx;
  OpBuilder b(&ctx);

  const auto _ = ShapedType::kDynamic;
  const auto f32 = b.getF32Type();
  auto strided = [&ctx](ArrayRef<int64_t> s) {
    return StridedLayoutAttr::get(&ctx, 0, s);
  };

  // memref<2x2x2xf32, strided<[4,2,1]>
  auto m1 = MemRefType::get({2, 2, 2}, f32, strided({4, 2, 1}));
  EXPECT_TRUE(m1.areTrailingDimsContiguous(1));
  EXPECT_TRUE(m1.areTrailingDimsContiguous(2));
  EXPECT_TRUE(m1.areTrailingDimsContiguous(3));

  // memref<2x2x2xf32, strided<[8,2,1]>
  auto m2 = MemRefType::get({2, 2, 2}, f32, strided({8, 2, 1}));
  EXPECT_TRUE(m2.areTrailingDimsContiguous(1));
  EXPECT_TRUE(m2.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m2.areTrailingDimsContiguous(3));

  // memref<2x2x2xf32, strided<[8,4,1]>
  auto m3 = MemRefType::get({2, 2, 2}, f32, strided({8, 4, 1}));
  EXPECT_TRUE(m3.areTrailingDimsContiguous(1));
  EXPECT_FALSE(m3.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m3.areTrailingDimsContiguous(3));

  // memref<2x2x2xf32, strided<[8,4,2]>
  auto m4 = MemRefType::get({2, 2, 2}, f32, strided({8, 4, 2}));
  EXPECT_FALSE(m4.areTrailingDimsContiguous(1));
  EXPECT_FALSE(m4.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m4.areTrailingDimsContiguous(3));

  // memref<2x2x?xf32, strided<[?,?,1]>
  auto m5 = MemRefType::get({2, 2, _}, f32, strided({_, _, 1}));
  EXPECT_TRUE(m5.areTrailingDimsContiguous(1));
  EXPECT_FALSE(m5.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m5.areTrailingDimsContiguous(3));

  // memref<2x2x?xf32, strided<[?,?,2]>
  auto m6 = MemRefType::get({2, 2, _}, f32, strided({_, _, 2}));
  EXPECT_FALSE(m6.areTrailingDimsContiguous(1));
  EXPECT_FALSE(m6.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m6.areTrailingDimsContiguous(3));

  // memref<2x?x2xf32, strided<[?,2,1]>
  auto m7 = MemRefType::get({2, _, 2}, f32, strided({_, 2, 1}));
  EXPECT_TRUE(m7.areTrailingDimsContiguous(1));
  EXPECT_TRUE(m7.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m7.areTrailingDimsContiguous(3));

  // memref<2x?x2xf32, strided<[?,4,1]>
  auto m8 = MemRefType::get({2, _, 2}, f32, strided({_, 4, 1}));
  EXPECT_TRUE(m8.areTrailingDimsContiguous(1));
  EXPECT_FALSE(m8.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m8.areTrailingDimsContiguous(3));

  // memref<2x?x2xf32, strided<[?,4,2]>
  auto m9 = MemRefType::get({2, _, 2}, f32, strided({_, 4, 2}));
  EXPECT_FALSE(m9.areTrailingDimsContiguous(1));
  EXPECT_FALSE(m9.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m9.areTrailingDimsContiguous(3));

  // memref<?x2x2xf32, strided<[4,2,1]>
  auto m10 = MemRefType::get({_, 2, 2}, f32, strided({4, 2, 1}));
  EXPECT_TRUE(m10.areTrailingDimsContiguous(1));
  EXPECT_TRUE(m10.areTrailingDimsContiguous(2));
  EXPECT_TRUE(m10.areTrailingDimsContiguous(3));

  // memref<?x2x2xf32, strided<[8,2,1]>
  auto m11 = MemRefType::get({_, 2, 2}, f32, strided({8, 2, 1}));
  EXPECT_TRUE(m11.areTrailingDimsContiguous(1));
  EXPECT_TRUE(m11.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m11.areTrailingDimsContiguous(3));

  // memref<?x2x2xf32, strided<[8,4,1]>
  auto m12 = MemRefType::get({_, 2, 2}, f32, strided({8, 4, 1}));
  EXPECT_TRUE(m12.areTrailingDimsContiguous(1));
  EXPECT_FALSE(m12.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m12.areTrailingDimsContiguous(3));

  // memref<?x2x2xf32, strided<[8,4,2]>
  auto m13 = MemRefType::get({_, 2, 2}, f32, strided({8, 4, 2}));
  EXPECT_FALSE(m13.areTrailingDimsContiguous(1));
  EXPECT_FALSE(m13.areTrailingDimsContiguous(2));
  EXPECT_FALSE(m13.areTrailingDimsContiguous(3));
}

TEST(MemRefLayout, identityMaps) {
  MLIRContext ctx;
  OpBuilder b(&ctx);

  const auto _ = ShapedType::kDynamic;
  const auto f32 = b.getF32Type();

  // memref<2x2x2xf32>
  auto m1 = MemRefType::get({2, 2, 2}, f32);
  EXPECT_EQ(m1.getNumContiguousTrailingDims(), 3);
  EXPECT_TRUE(m1.areTrailingDimsContiguous(1));
  EXPECT_TRUE(m1.areTrailingDimsContiguous(2));
  EXPECT_TRUE(m1.areTrailingDimsContiguous(3));

  // memref<?x?x?xf32>
  auto m2 = MemRefType::get({_, _, _}, f32);
  EXPECT_EQ(m2.getNumContiguousTrailingDims(), 3);
  EXPECT_TRUE(m2.areTrailingDimsContiguous(1));
  EXPECT_TRUE(m2.areTrailingDimsContiguous(2));
  EXPECT_TRUE(m2.areTrailingDimsContiguous(3));
}
