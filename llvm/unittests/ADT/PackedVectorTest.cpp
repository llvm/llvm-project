//===- llvm/unittest/ADT/PackedVectorTest.cpp - PackedVector tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// BitVectorTest tests fail on PowerPC for unknown reasons, so disable this
// as well since it depends on a BitVector.
#ifndef __ppc__

#include "llvm/ADT/PackedVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(PackedVectorTest, Operation) {
  PackedVector<unsigned, 2> Vec;
  EXPECT_EQ(0U, Vec.size());
  EXPECT_TRUE(Vec.empty());

  Vec.resize(5);
  EXPECT_EQ(5U, Vec.size());
  EXPECT_FALSE(Vec.empty());

  Vec.resize(11);
  EXPECT_EQ(11U, Vec.size());
  EXPECT_FALSE(Vec.empty());

  PackedVector<unsigned, 2> Vec2(3);
  EXPECT_EQ(3U, Vec2.size());
  EXPECT_FALSE(Vec2.empty());

  Vec.clear();
  EXPECT_EQ(0U, Vec.size());
  EXPECT_TRUE(Vec.empty());

  Vec.push_back(2);
  Vec.push_back(0);
  Vec.push_back(1);
  Vec.push_back(3);

  EXPECT_EQ(2U, Vec[0]);
  EXPECT_EQ(0U, Vec[1]);
  EXPECT_EQ(1U, Vec[2]);
  EXPECT_EQ(3U, Vec[3]);

  EXPECT_FALSE(Vec == Vec2);
  EXPECT_TRUE(Vec != Vec2);

  Vec = Vec2;
  EXPECT_TRUE(Vec == Vec2);
  EXPECT_FALSE(Vec != Vec2);

  Vec[1] = 1;
  Vec2[1] = 2;
  Vec |= Vec2;
  EXPECT_EQ(3U, Vec[1]);
}

#ifdef EXPECT_DEBUG_DEATH

TEST(PackedVectorTest, UnsignedValues) {
  PackedVector<unsigned, 2> Vec(1);
  Vec[0] = 0;
  Vec[0] = 1;
  Vec[0] = 2;
  Vec[0] = 3;
  EXPECT_DEBUG_DEATH(Vec[0] = 4, "value is too big");
  EXPECT_DEBUG_DEATH(Vec[0] = -1, "value is too big");
  EXPECT_DEBUG_DEATH(Vec[0] = 0x100, "value is too big");

  PackedVector<unsigned, 3> Vec2(1);
  Vec2[0] = 0;
  Vec2[0] = 7;
  EXPECT_DEBUG_DEATH(Vec[0] = 8, "value is too big");
}

TEST(PackedVectorTest, SignedValues) {
  PackedVector<signed, 2> Vec(1);
  Vec[0] = -2;
  Vec[0] = -1;
  Vec[0] = 0;
  Vec[0] = 1;
  EXPECT_DEBUG_DEATH(Vec[0] = -3, "value is too big");
  EXPECT_DEBUG_DEATH(Vec[0] = 2, "value is too big");

  PackedVector<signed, 3> Vec2(1);
  Vec2[0] = -4;
  Vec2[0] = 3;
  EXPECT_DEBUG_DEATH(Vec[0] = -5, "value is too big");
  EXPECT_DEBUG_DEATH(Vec[0] = 4, "value is too big");
}

#endif

}

#endif
