//===- MemRefTypeTest.cpp - MemRefType unit tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

namespace {

TEST(MemRefTypeTest, GetStridesAndOffset) {
  MLIRContext context;

  SmallVector<int64_t> shape({2, 3, 4});
  Type f32 = FloatType::getF32(&context);

  AffineMap map1 = makeStridedLinearLayoutMap({12, 4, 1}, 5, &context);
  MemRefType type1 = MemRefType::get(shape, f32, {map1});
  SmallVector<int64_t> strides1;
  int64_t offset1 = -1;
  LogicalResult res1 = getStridesAndOffset(type1, strides1, offset1);
  ASSERT_TRUE(res1.succeeded());
  ASSERT_EQ(3u, strides1.size());
  EXPECT_EQ(12, strides1[0]);
  EXPECT_EQ(4, strides1[1]);
  EXPECT_EQ(1, strides1[2]);
  ASSERT_EQ(5, offset1);

  AffineMap map2 = AffineMap::getPermutationMap({1, 2, 0}, &context);
  AffineMap map3 = makeStridedLinearLayoutMap({8, 2, 1}, 0, &context);
  MemRefType type2 = MemRefType::get(shape, f32, {map2, map3});
  SmallVector<int64_t> strides2;
  int64_t offset2 = -1;
  LogicalResult res2 = getStridesAndOffset(type2, strides2, offset2);
  ASSERT_TRUE(res2.succeeded());
  ASSERT_EQ(3u, strides2.size());
  EXPECT_EQ(1, strides2[0]);
  EXPECT_EQ(8, strides2[1]);
  EXPECT_EQ(2, strides2[2]);
  ASSERT_EQ(0, offset2);
}

} // end namespace
