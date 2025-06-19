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

  // Create a sequence of test cases, starting with the base case of a
  // contiguous 2x2x2 memref with fixed dimensions and then at each step
  // introducing one dynamic dimension starting from the right.
  // With thus obtained memref, start with maximally contiguous strides
  // and then at each step gradually introduce discontinuity by increasing
  // a fixed stride size from the left to right.

  // In these and the following test cases the intent is to achieve code
  // coverage of the main loop in `MemRefType::getNumContiguousTrailingDims()`.

  // memref<2x2x2xf32, strided<[4,2,1]>>
  auto m1 = MemRefType::get({2, 2, 2}, f32, strided({4, 2, 1}));
  EXPECT_EQ(m1.getNumContiguousTrailingDims(), 3);

  // memref<2x2x2xf32, strided<[8,2,1]>>
  auto m2 = MemRefType::get({2, 2, 2}, f32, strided({8, 2, 1}));
  EXPECT_EQ(m2.getNumContiguousTrailingDims(), 2);

  // memref<2x2x2xf32, strided<[8,4,1]>>
  auto m3 = MemRefType::get({2, 2, 2}, f32, strided({8, 4, 1}));
  EXPECT_EQ(m3.getNumContiguousTrailingDims(), 1);

  // memref<2x2x2xf32, strided<[8,4,2]>>
  auto m4 = MemRefType::get({2, 2, 2}, f32, strided({8, 4, 2}));
  EXPECT_EQ(m4.getNumContiguousTrailingDims(), 0);

  // memref<2x2x?xf32, strided<[?,?,1]>>
  auto m5 = MemRefType::get({2, 2, _}, f32, strided({_, _, 1}));
  EXPECT_EQ(m5.getNumContiguousTrailingDims(), 1);

  // memref<2x2x?xf32, strided<[?,?,2]>>
  auto m6 = MemRefType::get({2, 2, _}, f32, strided({_, _, 2}));
  EXPECT_EQ(m6.getNumContiguousTrailingDims(), 0);

  // memref<2x?x2xf32, strided<[?,2,1]>>
  auto m7 = MemRefType::get({2, _, 2}, f32, strided({_, 2, 1}));
  EXPECT_EQ(m7.getNumContiguousTrailingDims(), 2);

  // memref<2x?x2xf32, strided<[?,4,1]>>
  auto m8 = MemRefType::get({2, _, 2}, f32, strided({_, 4, 1}));
  EXPECT_EQ(m8.getNumContiguousTrailingDims(), 1);

  // memref<2x?x2xf32, strided<[?,4,2]>>
  auto m9 = MemRefType::get({2, _, 2}, f32, strided({_, 4, 2}));
  EXPECT_EQ(m9.getNumContiguousTrailingDims(), 0);

  // memref<?x2x2xf32, strided<[4,2,1]>>
  auto m10 = MemRefType::get({_, 2, 2}, f32, strided({4, 2, 1}));
  EXPECT_EQ(m10.getNumContiguousTrailingDims(), 3);

  // memref<?x2x2xf32, strided<[8,2,1]>>
  auto m11 = MemRefType::get({_, 2, 2}, f32, strided({8, 2, 1}));
  EXPECT_EQ(m11.getNumContiguousTrailingDims(), 2);

  // memref<?x2x2xf32, strided<[8,4,1]>>
  auto m12 = MemRefType::get({_, 2, 2}, f32, strided({8, 4, 1}));
  EXPECT_EQ(m12.getNumContiguousTrailingDims(), 1);

  // memref<?x2x2xf32, strided<[8,4,2]>>
  auto m13 = MemRefType::get({_, 2, 2}, f32, strided({8, 4, 2}));
  EXPECT_EQ(m13.getNumContiguousTrailingDims(), 0);

  //
  // Repeat a similar process, but this time introduce a unit memref dimension
  // to test that strides corresponding to unit dimensions are immaterial, even
  // if dynamic.
  //

  // memref<2x2x1xf32, strided<[2,1,2]>>
  auto m14 = MemRefType::get({2, 2, 1}, f32, strided({2, 1, 2}));
  EXPECT_EQ(m14.getNumContiguousTrailingDims(), 3);

  // memref<2x2x1xf32, strided<[2,1,?]>>
  auto m15 = MemRefType::get({2, 2, 1}, f32, strided({2, 1, _}));
  EXPECT_EQ(m15.getNumContiguousTrailingDims(), 3);

  // memref<2x2x1xf32, strided<[4,2,2]>>
  auto m16 = MemRefType::get({2, 2, 1}, f32, strided({4, 2, 2}));
  EXPECT_EQ(m16.getNumContiguousTrailingDims(), 1);

  // memref<2x1x2xf32, strided<[2,4,1]>>
  auto m17 = MemRefType::get({2, 1, 2}, f32, strided({2, 4, 1}));
  EXPECT_EQ(m17.getNumContiguousTrailingDims(), 3);

  // memref<2x1x2xf32, strided<[2,?,1]>>
  auto m18 = MemRefType::get({2, 1, 2}, f32, strided({2, _, 1}));
  EXPECT_EQ(m18.getNumContiguousTrailingDims(), 3);

  //
  // Special case for identity maps and no explicit `strided` attribute - the
  // memref is entirely contiguous even if the strides cannot be determined
  // statically.
  //

  // memref<?x?x?xf32>
  auto m19 = MemRefType::get({_, _, _}, f32);
  EXPECT_EQ(m19.getNumContiguousTrailingDims(), 3);
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

  // Pick up a random test case among the ones already present in the file and
  // ensure `areTrailingDimsContiguous(k)` returns `true` up to the value
  // returned by `getNumContiguousTrailingDims` and `false` from that point on
  // up to the memref rank.

  // memref<2x?x2xf32, strided<[?,2,1]>>
  auto m = MemRefType::get({2, _, 2}, f32, strided({_, 2, 1}));
  int64_t n = m.getNumContiguousTrailingDims();
  for (int64_t i = 0; i <= n; ++i)
    EXPECT_TRUE(m.areTrailingDimsContiguous(i));

  int64_t r = m.getRank();
  for (int64_t i = n + 1; i <= r; ++i)
    EXPECT_FALSE(m.areTrailingDimsContiguous(i));
}
