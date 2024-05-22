//===- IndexingUtilsTest.cpp - IndexingUtils unit tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(StaticTileOffsetRange, checkIteratorCanonicalOrder) {
  // Tile <4x8> by <2x4> with canonical row-major order.
  std::vector<SmallVector<int64_t>> expected = {{0, 0}, {0, 4}, {2, 0}, {2, 4}};
  for (auto [idx, tileOffset] :
       llvm::enumerate(StaticTileOffsetRange({4, 8}, {2, 4}, {0, 1})))
    EXPECT_EQ(tileOffset, expected[idx]);

  // Check the constructor for default order and test use with zip iterator.
  for (auto [tileOffset, tileOffsetDefault] :
       llvm::zip(StaticTileOffsetRange({4, 8}, {2, 4}, {0, 1}),
                 StaticTileOffsetRange({4, 8}, {2, 4})))
    EXPECT_EQ(tileOffset, tileOffsetDefault);
}

TEST(StaticTileOffsetRange, checkIteratorRowMajorOrder) {
  // Tile <4x8> by <2x4> with canonical row-major order.
  std::vector<SmallVector<int64_t>> expected = {{0, 0}, {2, 0}, {0, 4}, {2, 4}};
  for (auto [idx, tileOffset] :
       llvm::enumerate(StaticTileOffsetRange({4, 8}, {2, 4}, {1, 0})))
    EXPECT_EQ(tileOffset, expected[idx]);
}

TEST(StaticTileOffsetRange, checkLeadingOneFill) {
  // Tile <4x8> by <4>. A smaller tile shape gets right-aligned to the shape.
  for (auto [idx, tileOffset] :
       llvm::enumerate(StaticTileOffsetRange({4, 8}, {4}))) {
    SmallVector<int64_t> expected = {static_cast<int64_t>(idx) / 2,
                                     static_cast<int64_t>(idx) % 2 * 4};
    EXPECT_EQ(tileOffset, expected);
  }
  for (auto [idx, tileOffset] :
       llvm::enumerate(StaticTileOffsetRange({1, 4, 8}, {4}, {2, 1, 0}))) {
    SmallVector<int64_t> expected = {0, static_cast<int64_t>(idx) % 4,
                                     (static_cast<int64_t>(idx) / 4) * 4};
    EXPECT_EQ(tileOffset, expected);
  }
}

TEST(StaticTileOffsetRange, checkIterator3DPermutation) {
  // Tile <8x4x2> by <4x2x1> with permutation [1, 0, 2]
  for (auto [idx, tileOffset] : llvm::enumerate(
           StaticTileOffsetRange({8, 4, 2}, {4, 2, 1}, {1, 0, 2}))) {
    SmallVector<int64_t> expected = {((static_cast<int64_t>(idx) / 2) % 2) * 4,
                                     ((static_cast<int64_t>(idx) / 4) % 2) * 2,
                                     static_cast<int64_t>(idx) % 2};
    EXPECT_EQ(tileOffset, expected);
  }

  // Tile <10x20x30> by <5x10x16> with permutation [2, 0, 1]
  for (auto [idx, tileOffset] : llvm::enumerate(
           StaticTileOffsetRange({10, 20, 30}, {5, 10, 15}, {2, 0, 1}))) {
    SmallVector<int64_t> expected = {((static_cast<int64_t>(idx) / 2) % 2) * 5,
                                     (static_cast<int64_t>(idx) % 2) * 10,
                                     (static_cast<int64_t>(idx) / 4) % 2 * 15};
    EXPECT_EQ(tileOffset, expected);
  }
}
