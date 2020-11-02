//===----------- ImmutableMapTest.cpp - ImmutableMap unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ImmutableMap.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ImmutableMapTest, EmptyIntMapTest) {
  ImmutableMap<int, int>::Factory f;

  EXPECT_TRUE(f.getEmptyMap() == f.getEmptyMap());
  EXPECT_FALSE(f.getEmptyMap() != f.getEmptyMap());
  EXPECT_TRUE(f.getEmptyMap().isEmpty());

  ImmutableMap<int, int> S = f.getEmptyMap();
  EXPECT_EQ(0u, S.getHeight());
  EXPECT_TRUE(S.begin() == S.end());
  EXPECT_FALSE(S.begin() != S.end());
}

TEST(ImmutableMapTest, MultiElemIntMapTest) {
  ImmutableMap<int, int>::Factory f;
  ImmutableMap<int, int> S = f.getEmptyMap();

  ImmutableMap<int, int> S2 = f.add(f.add(f.add(S, 3, 10), 4, 11), 5, 12);

  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S2.isEmpty());

  EXPECT_EQ(nullptr, S.lookup(3));
  EXPECT_EQ(nullptr, S.lookup(9));

  EXPECT_EQ(10, *S2.lookup(3));
  EXPECT_EQ(11, *S2.lookup(4));
  EXPECT_EQ(12, *S2.lookup(5));

  EXPECT_EQ(5, S2.getMaxElement()->first);
  EXPECT_EQ(3U, S2.getHeight());
}

TEST(ImmutableMapTest, EmptyIntMapRefTest) {
  using int_int_map = ImmutableMapRef<int, int>;
  ImmutableMapRef<int, int>::FactoryTy f;

  EXPECT_TRUE(int_int_map::getEmptyMap(&f) == int_int_map::getEmptyMap(&f));
  EXPECT_FALSE(int_int_map::getEmptyMap(&f) != int_int_map::getEmptyMap(&f));
  EXPECT_TRUE(int_int_map::getEmptyMap(&f).isEmpty());

  int_int_map S = int_int_map::getEmptyMap(&f);
  EXPECT_EQ(0u, S.getHeight());
  EXPECT_TRUE(S.begin() == S.end());
  EXPECT_FALSE(S.begin() != S.end());
}

TEST(ImmutableMapTest, MultiElemIntMapRefTest) {
  ImmutableMapRef<int, int>::FactoryTy f;

  ImmutableMapRef<int, int> S = ImmutableMapRef<int, int>::getEmptyMap(&f);

  ImmutableMapRef<int, int> S2 = S.add(3, 10).add(4, 11).add(5, 12);

  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S2.isEmpty());

  EXPECT_EQ(nullptr, S.lookup(3));
  EXPECT_EQ(nullptr, S.lookup(9));

  EXPECT_EQ(10, *S2.lookup(3));
  EXPECT_EQ(11, *S2.lookup(4));
  EXPECT_EQ(12, *S2.lookup(5));

  EXPECT_EQ(5, S2.getMaxElement()->first);
  EXPECT_EQ(3U, S2.getHeight());
}

  TEST(ImmutableMapTest, MapOfMapRefsTest) {
  ImmutableMap<int, ImmutableMapRef<int, int>>::Factory f;

  EXPECT_TRUE(f.getEmptyMap() == f.getEmptyMap());
  }

}
