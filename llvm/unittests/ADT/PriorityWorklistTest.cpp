//===- llvm/unittest/ADT/PriorityWorklist.cpp -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PriorityWorklist unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PriorityWorklist.h"
#include "gtest/gtest.h"

namespace {

using namespace llvm;

template <typename T> class PriorityWorklistTest : public ::testing::Test {};
typedef ::testing::Types<PriorityWorklist<int>, SmallPriorityWorklist<int, 2>>
    TestTypes;
TYPED_TEST_CASE(PriorityWorklistTest, TestTypes);

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
