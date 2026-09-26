//===- llvm/unittest/ADT/SetVector.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SetVector unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(SetVector, EraseTest) {
  SetVector<int> S;
  S.insert(0);
  S.insert(1);
  S.insert(2);

  auto I = S.erase(std::next(S.begin()));

  // Test that the returned iterator is the expected one-after-erase
  // and the size/contents is the expected sequence {0, 2}.
  EXPECT_EQ(std::next(S.begin()), I);
  EXPECT_EQ(2u, S.size());
  EXPECT_EQ(0, *S.begin());
  EXPECT_EQ(2, *std::next(S.begin()));
}

TEST(SetVector, ContainsTest) {
  SetVector<int> S;
  S.insert(0);
  S.insert(1);
  S.insert(2);

  EXPECT_TRUE(S.contains(0));
  EXPECT_TRUE(S.contains(1));
  EXPECT_TRUE(S.contains(2));
  EXPECT_FALSE(S.contains(-1));

  S.insert(2);
  EXPECT_TRUE(S.contains(2));

  S.remove(2);
  EXPECT_FALSE(S.contains(2));
}

TEST(SetVector, ConstPtrKeyTest) {
  SetVector<int *> S, T;
  int i, j, k, m, n;

  S.insert(&i);
  S.insert(&j);
  S.insert(&k);

  EXPECT_TRUE(S.contains(&i));
  EXPECT_TRUE(S.contains(&j));
  EXPECT_TRUE(S.contains(&k));

  EXPECT_TRUE(S.contains((const int *)&i));
  EXPECT_TRUE(S.contains((const int *)&j));
  EXPECT_TRUE(S.contains((const int *)&k));

  EXPECT_TRUE(S.contains(S[0]));
  EXPECT_TRUE(S.contains(S[1]));
  EXPECT_TRUE(S.contains(S[2]));

  S.remove(&k);
  EXPECT_FALSE(S.contains(&k));
  EXPECT_FALSE(S.contains((const int *)&k));

  T.insert(&j);
  T.insert(&m);
  T.insert(&n);

  EXPECT_TRUE(S.set_union(T));
  EXPECT_TRUE(S.contains(&m));
  EXPECT_TRUE(S.contains((const int *)&m));

  S.set_subtract(T);
  EXPECT_FALSE(S.contains(&j));
  EXPECT_FALSE(S.contains((const int *)&j));
}

TEST(SetVector, CtorRange) {
  constexpr unsigned Args[] = {3, 1, 2};
  SetVector<unsigned> Set(llvm::from_range, Args);
  EXPECT_THAT(Set, ::testing::ElementsAre(3, 1, 2));
}

TEST(SetVector, InsertRange) {
  SetVector<unsigned> Set;
  constexpr unsigned Args[] = {3, 1, 2};
  Set.insert_range(Args);
  EXPECT_THAT(Set, ::testing::ElementsAre(3, 1, 2));
}

TEST(SmallSetVector, CtorRange) {
  constexpr unsigned Args[] = {3, 1, 2};
  SmallSetVector<unsigned, 4> Set(llvm::from_range, Args);
  EXPECT_THAT(Set, ::testing::ElementsAre(3, 1, 2));
}

TEST(SetVector, CtorInitList) {
  SetVector<unsigned> Set = {3, 1, 2, 1};
  EXPECT_THAT(Set, ::testing::ElementsAre(3, 1, 2));
}

TEST(SmallSetVector, CtorInitList) {
  SmallSetVector<unsigned, 4> Set = {3, 1, 2, 1};
  EXPECT_THAT(Set, ::testing::ElementsAre(3, 1, 2));
}

TEST(SmallSetVectorImpl, TypeErasedReference) {
  auto Populate = [](SmallSetVectorImpl<unsigned> &S) {
    S.insert(3);
    S.insert(1);
    S.insert(2);
    S.insert(1);
    S.insert(4);
    S.insert(5);
  };

  SmallSetVector<unsigned, 2> Small2;
  Populate(Small2);
  EXPECT_THAT(Small2, ::testing::ElementsAre(3, 1, 2, 4, 5));

  SmallSetVector<unsigned, 16> Small16;
  Populate(Small16);
  EXPECT_THAT(Small16, ::testing::ElementsAre(3, 1, 2, 4, 5));

  SetVector<unsigned> DefaultSV;
  Populate(DefaultSV);
  EXPECT_THAT(DefaultSV, ::testing::ElementsAre(3, 1, 2, 4, 5));

  // Cross-capacity copy and move construction/assignment via SmallSetVectorImpl
  SmallSetVector<unsigned, 4> CopyFrom16(Small16);
  EXPECT_THAT(CopyFrom16, ::testing::ElementsAre(3, 1, 2, 4, 5));

  SmallSetVector<unsigned, 8> MoveFrom2(std::move(Small2));
  EXPECT_THAT(MoveFrom2, ::testing::ElementsAre(3, 1, 2, 4, 5));
}
