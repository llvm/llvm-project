//===----------- ImmutableMapTest.cpp - ImmutableMap unit tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

}
