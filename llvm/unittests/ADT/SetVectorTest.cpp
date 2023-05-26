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
  SetVector<int *, SmallVector<int *, 8>, SmallPtrSet<const int *, 8>> S, T;
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
