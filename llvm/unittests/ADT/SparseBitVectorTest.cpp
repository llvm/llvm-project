//===- llvm/unittest/ADT/SparseBitVectorTest.cpp - SparseBitVector tests --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SparseBitVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(SparseBitVectorTest, TrivialOperation) {
  SparseBitVector<> Vec;
  EXPECT_EQ(0U, Vec.count());
  EXPECT_FALSE(Vec.test(17));
  Vec.set(5);
  EXPECT_TRUE(Vec.test(5));
  EXPECT_FALSE(Vec.test(17));
  Vec.reset(6);
  EXPECT_TRUE(Vec.test(5));
  EXPECT_FALSE(Vec.test(6));
  Vec.reset(5);
  EXPECT_FALSE(Vec.test(5));
  EXPECT_TRUE(Vec.test_and_set(17));
  EXPECT_FALSE(Vec.test_and_set(17));
  EXPECT_TRUE(Vec.test(17));
  Vec.clear();
  EXPECT_FALSE(Vec.test(17));
}

TEST(SparseBitVectorTest, IntersectWith) {
  SparseBitVector<> Vec, Other;

  Vec.set(1);
  Other.set(1);
  EXPECT_FALSE(Vec &= Other);
  EXPECT_TRUE(Vec.test(1));

  Vec.clear();
  Vec.set(5);
  Other.clear();
  Other.set(6);
  EXPECT_TRUE(Vec &= Other);
  EXPECT_TRUE(Vec.empty());

  Vec.clear();
  Vec.set(5);
  Other.clear();
  Other.set(225);
  EXPECT_TRUE(Vec &= Other);
  EXPECT_TRUE(Vec.empty());

  Vec.clear();
  Vec.set(225);
  Other.clear();
  Other.set(5);
  EXPECT_TRUE(Vec &= Other);
  EXPECT_TRUE(Vec.empty());
}

TEST(SparseBitVectorTest, SelfAssignment) {
  SparseBitVector<> Vec, Other;

  Vec.set(23);
  Vec.set(234);
  Vec = static_cast<SparseBitVector<> &>(Vec);
  EXPECT_TRUE(Vec.test(23));
  EXPECT_TRUE(Vec.test(234));

  Vec.clear();
  Vec.set(17);
  Vec.set(256);
  EXPECT_FALSE(Vec |= Vec);
  EXPECT_TRUE(Vec.test(17));
  EXPECT_TRUE(Vec.test(256));

  Vec.clear();
  Vec.set(56);
  Vec.set(517);
  EXPECT_FALSE(Vec &= Vec);
  EXPECT_TRUE(Vec.test(56));
  EXPECT_TRUE(Vec.test(517));

  Vec.clear();
  Vec.set(99);
  Vec.set(333);
  EXPECT_TRUE(Vec.intersectWithComplement(Vec));
  EXPECT_TRUE(Vec.empty());
  EXPECT_FALSE(Vec.intersectWithComplement(Vec));

  Vec.clear();
  Vec.set(28);
  Vec.set(43);
  Vec.intersectWithComplement(Vec, Vec);
  EXPECT_TRUE(Vec.empty());

  Vec.clear();
  Vec.set(42);
  Vec.set(567);
  Other.set(55);
  Other.set(567);
  Vec.intersectWithComplement(Vec, Other);
  EXPECT_TRUE(Vec.test(42));
  EXPECT_FALSE(Vec.test(567));

  Vec.clear();
  Vec.set(19);
  Vec.set(21);
  Other.clear();
  Other.set(19);
  Other.set(31);
  Vec.intersectWithComplement(Other, Vec);
  EXPECT_FALSE(Vec.test(19));
  EXPECT_TRUE(Vec.test(31));

  Vec.clear();
  Vec.set(1);
  Other.clear();
  Other.set(59);
  Other.set(75);
  Vec.intersectWithComplement(Other, Other);
  EXPECT_TRUE(Vec.empty());
}

TEST(SparseBitVectorTest, Find) {
  SparseBitVector<> Vec;
  Vec.set(1);
  EXPECT_EQ(1, Vec.find_first());
  EXPECT_EQ(1, Vec.find_last());

  Vec.set(2);
  EXPECT_EQ(1, Vec.find_first());
  EXPECT_EQ(2, Vec.find_last());

  Vec.set(0);
  Vec.set(3);
  EXPECT_EQ(0, Vec.find_first());
  EXPECT_EQ(3, Vec.find_last());

  Vec.reset(1);
  Vec.reset(0);
  Vec.reset(3);
  EXPECT_EQ(2, Vec.find_first());
  EXPECT_EQ(2, Vec.find_last());

  // Set some large bits to ensure we are pulling bits from more than just a
  // single bitword.
  Vec.set(500);
  Vec.set(2000);
  Vec.set(3000);
  Vec.set(4000);
  Vec.reset(2);
  EXPECT_EQ(500, Vec.find_first());
  EXPECT_EQ(4000, Vec.find_last());

  Vec.reset(500);
  Vec.reset(3000);
  Vec.reset(4000);
  EXPECT_EQ(2000, Vec.find_first());
  EXPECT_EQ(2000, Vec.find_last());

  Vec.clear();
}
}
