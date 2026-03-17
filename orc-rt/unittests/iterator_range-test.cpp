//===- iterator_range-test.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's iterator_range.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/iterator_range.h"
#include "gtest/gtest.h"

#include <type_traits>
#include <vector>

using namespace orc_rt;

TEST(IteratorRangeTest, EmptyArray) {
  int A[1]; // zero-length arrays aren't allowed.
  iterator_range<int *> R(std::begin(A), std::begin(A));

  EXPECT_TRUE(R.empty());
  EXPECT_EQ(R.begin(), R.end());
}

TEST(IteratorRangeTest, NonEmptyArray) {
  int A[] = {10, 11, 12, 13, 14, 15};

  size_t Index = 0;
  for (auto &E : iterator_range(A))
    EXPECT_EQ(E, A[Index++]);
}

TEST(IteratorRangeTest, EmptyVector) {
  std::vector<int> V;
  auto R = iterator_range(V);
  EXPECT_TRUE(R.empty());
  EXPECT_EQ(R.begin(), R.end());
}

TEST(IteratorRangeTest, NonEmptyVector) {
  std::vector<int> V({{10, 12, 14, 16, 18, 20}});

  size_t Index = 0;
  for (auto &E : iterator_range(V))
    EXPECT_EQ(E, V[Index++]);
}

TEST(IteratorRangeTest, Subrange) {
  std::vector<int> V = {1, 2, 3, 4, 5};
  iterator_range R(V.begin() + 1, V.begin() + 4);

  EXPECT_FALSE(R.empty());
  std::vector<int> Result(R.begin(), R.end());
  EXPECT_EQ(Result, (std::vector<int>{2, 3, 4}));
}

TEST(IteratorRangeTest, MutateThroughRange) {
  std::vector<int> V = {1, 2, 3};
  for (auto &E : iterator_range(V))
    E *= 2;
  EXPECT_EQ(V, (std::vector<int>{2, 4, 6}));
}

TEST(IteratorRangeTest, NonEmptyIsNotEmpty) {
  std::vector<int> V = {1};
  auto R = iterator_range(V);
  EXPECT_FALSE(R.empty());
}

TEST(IteratorRangeTest, ConstContainer) {
  const std::vector<int> V = {1, 2, 3};
  for (auto &E : iterator_range(V))
    static_assert(std::is_const_v<std::remove_reference_t<decltype(E)>>,
                  "elements from const container should be const");
}
