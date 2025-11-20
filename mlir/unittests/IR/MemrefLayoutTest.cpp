//===- LayoutTest.cpp - unit tests related to memref layout ---------------===//
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

//
// Test the correctness of `memref::getNumContiguousTrailingDims`
//
TEST(MemRefLayout, numContigDim) {
  MLIRContext ctx;
  OpBuilder b(&ctx);

  const int64_t _ = ShapedType::kDynamic;
  const FloatType f32 = b.getF32Type();
  auto strided = [&ctx](ArrayRef<int64_t> s) {
    return StridedLayoutAttr::get(&ctx, 0, s);
  };

  // Special case for identity maps and no explicit `strided` attribute - the
  // memref is entirely contiguous even if the strides cannot be determined
  // statically.

  // memref<?x?x?xf32>
  auto m0 = MemRefType::get({_, _, _}, f32);
  EXPECT_EQ(m0.getNumContiguousTrailingDims(), 3);

  // Conservatively assume memref is sparse everywhere if cannot get the
  // strides.

  // memref<2x2x2xf32, (i,j,k)->(i,k,j)>
  auto m1 = MemRefType::get(
      {2, 2, 2}, f32,
      AffineMap::getPermutationMap(ArrayRef<int64_t>{0, 2, 1}, &ctx));
  EXPECT_EQ(m1.getNumContiguousTrailingDims(), 0);

  // A base cases of a fixed memref with the usual strides.

  // memref<2x2x2xf32, strided<[4, 2, 1]>>
  auto m3 = MemRefType::get({2, 2, 2}, f32, strided({4, 2, 1}));
  EXPECT_EQ(m3.getNumContiguousTrailingDims(), 3);

  // A fixed memref with a discontinuity in the rightmost dimension.

  // memref<2x2x2xf32, strided<[8, 4, 2]>>
  auto m4 = MemRefType::get({2, 2, 2}, f32, strided({8, 4, 2}));
  EXPECT_EQ(m4.getNumContiguousTrailingDims(), 0);

  // A fixed memref with a discontinuity in the "middle".

  // memref<2x2x2xf32, strided<[8, 2, 1]>>
  auto m5 = MemRefType::get({2, 2, 2}, f32, strided({8, 2, 1}));
  EXPECT_EQ(m5.getNumContiguousTrailingDims(), 2);

  // A dynamic memref where the dynamic dimension breaks continuity.

  // memref<2x?x2xf32, strided<[4, 2, 1]>>
  auto m6 = MemRefType::get({2, _, 2}, f32, strided({4, 2, 1}));
  EXPECT_EQ(m6.getNumContiguousTrailingDims(), 2);

  // A edge case of a dynamic memref where the dynamic dimension is the first
  // one.

  // memref<?x2x2xf32, strided<[4, 2, 1]>>
  auto m7 = MemRefType::get({2, _, 2}, f32, strided({4, 2, 1}));
  EXPECT_EQ(m7.getNumContiguousTrailingDims(), 2);

  // A memref with a unit dimension. Unit dimensions do not affect continuity,
  // even if the corresponding stride is dynamic.

  // memref<2x1x2xf32, strided<[2,?,1]>>
  auto m8 = MemRefType::get({2, 1, 2}, f32, strided({2, _, 1}));
  EXPECT_EQ(m8.getNumContiguousTrailingDims(), 3);
}

//
// Test the member function `memref::areTrailingDimsContiguous`
//
TEST(MemRefLayout, contigTrailingDim) {
  MLIRContext ctx;
  OpBuilder b(&ctx);

  const int64_t _ = ShapedType::kDynamic;
  const FloatType f32 = b.getF32Type();
  auto strided = [&ctx](ArrayRef<int64_t> s) {
    return StridedLayoutAttr::get(&ctx, 0, s);
  };

  // A not-entirely-continuous, not-entirely-discontinuous memref.
  // ensure `areTrailingDimsContiguous` returns `true` for the value
  // returned by `getNumContiguousTrailingDims` and `false` for the next bigger
  // number.

  // memref<2x?x2xf32, strided<[?,2,1]>>
  auto m = MemRefType::get({2, _, 2}, f32, strided({_, 2, 1}));
  int64_t n = m.getNumContiguousTrailingDims();
  EXPECT_TRUE(m.areTrailingDimsContiguous(n));
  ASSERT_TRUE(n + 1 <= m.getRank());
  EXPECT_FALSE(m.areTrailingDimsContiguous(n + 1));
}
