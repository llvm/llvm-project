//===- llvm/unittest/ADT/PriorityWorklist.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PriorityWorklist unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PriorityWorklist.h"
#include "gtest/gtest.h"
#include <list>
#include <vector>

namespace {

using namespace llvm;

template <typename T> class PriorityWorklistTest : public ::testing::Test {};
typedef ::testing::Types<PriorityWorklist<int>, SmallPriorityWorklist<int, 2>>
    TestTypes;
TYPED_TEST_SUITE(PriorityWorklistTest, TestTypes, );

TYPED_TEST(PriorityWorklistTest, Basic) {
  TypeParam W;
  EXPECT_TRUE(W.empty());
  EXPECT_EQ(0u, W.size());
  EXPECT_FALSE(W.count(42));

  EXPECT_TRUE(W.insert(21));
  EXPECT_TRUE(W.insert(42));
  EXPECT_TRUE(W.insert(17));

  EXPECT_FALSE(W.empty());
  EXPECT_EQ(3u, W.size());
  EXPECT_TRUE(W.count(42));

  EXPECT_FALSE(W.erase(75));
  EXPECT_EQ(3u, W.size());
  EXPECT_EQ(17, W.back());

  EXPECT_TRUE(W.erase(17));
  EXPECT_FALSE(W.count(17));
  EXPECT_EQ(2u, W.size());
  EXPECT_EQ(42, W.back());

  W.clear();
  EXPECT_TRUE(W.empty());
  EXPECT_EQ(0u, W.size());

  EXPECT_TRUE(W.insert(21));
  EXPECT_TRUE(W.insert(42));
  EXPECT_TRUE(W.insert(12));
  EXPECT_TRUE(W.insert(17));
  EXPECT_TRUE(W.count(12));
  EXPECT_TRUE(W.count(17));
  EXPECT_EQ(4u, W.size());
  EXPECT_EQ(17, W.back());
  EXPECT_TRUE(W.erase(12));
  EXPECT_FALSE(W.count(12));
  EXPECT_TRUE(W.count(17));
  EXPECT_EQ(3u, W.size());
  EXPECT_EQ(17, W.back());

  EXPECT_FALSE(W.insert(42));
  EXPECT_EQ(3u, W.size());
  EXPECT_EQ(42, W.pop_back_val());
  EXPECT_EQ(17, W.pop_back_val());
  EXPECT_EQ(21, W.pop_back_val());
  EXPECT_TRUE(W.empty());
}

TYPED_TEST(PriorityWorklistTest, InsertSequence) {
  TypeParam W;
  ASSERT_TRUE(W.insert(2));
  ASSERT_TRUE(W.insert(4));
  ASSERT_TRUE(W.insert(7));
  // Insert a sequence that has internal duplicates and a duplicate among
  // existing entries.
  W.insert(std::vector<int>({42, 13, 42, 7, 8}));
  EXPECT_EQ(8, W.pop_back_val());
  EXPECT_EQ(7, W.pop_back_val());
  EXPECT_EQ(42, W.pop_back_val());
  EXPECT_EQ(13, W.pop_back_val());
  EXPECT_EQ(4, W.pop_back_val());
  EXPECT_EQ(2, W.pop_back_val());
  ASSERT_TRUE(W.empty());

  // Simpler tests with various other input types.
  ASSERT_TRUE(W.insert(2));
  ASSERT_TRUE(W.insert(7));
  // Use a non-random-access container.
  W.insert(std::list<int>({7, 5}));
  EXPECT_EQ(5, W.pop_back_val());
  EXPECT_EQ(7, W.pop_back_val());
  EXPECT_EQ(2, W.pop_back_val());
  ASSERT_TRUE(W.empty());

  ASSERT_TRUE(W.insert(2));
  ASSERT_TRUE(W.insert(7));
  // Use a raw array.
  int A[] = {7, 5};
  W.insert(A);
  EXPECT_EQ(5, W.pop_back_val());
  EXPECT_EQ(7, W.pop_back_val());
  EXPECT_EQ(2, W.pop_back_val());
  ASSERT_TRUE(W.empty());

  ASSERT_TRUE(W.insert(2));
  ASSERT_TRUE(W.insert(7));
  // Inserting an empty sequence does nothing.
  W.insert(std::vector<int>());
  EXPECT_EQ(7, W.pop_back_val());
  EXPECT_EQ(2, W.pop_back_val());
  ASSERT_TRUE(W.empty());
}

TYPED_TEST(PriorityWorklistTest, EraseIf) {
  TypeParam W;
  W.insert(23);
  W.insert(10);
  W.insert(47);
  W.insert(42);
  W.insert(23);
  W.insert(13);
  W.insert(26);
  W.insert(42);
  EXPECT_EQ(6u, W.size());

  EXPECT_FALSE(W.erase_if([](int i) { return i > 100; }));
  EXPECT_EQ(6u, W.size());
  EXPECT_EQ(42, W.back());

  EXPECT_TRUE(W.erase_if([](int i) {
    assert(i != 0 && "Saw a null value!");
    return (i & 1) == 0;
  }));
  EXPECT_EQ(3u, W.size());
  EXPECT_FALSE(W.count(42));
  EXPECT_FALSE(W.count(26));
  EXPECT_FALSE(W.count(10));
  EXPECT_FALSE(W.insert(47));
  EXPECT_FALSE(W.insert(23));
  EXPECT_EQ(23, W.pop_back_val());
  EXPECT_EQ(47, W.pop_back_val());
  EXPECT_EQ(13, W.pop_back_val());
}

}
