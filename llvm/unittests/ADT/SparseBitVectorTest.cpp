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

}
