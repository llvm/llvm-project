//===- llvm/unittest/IR/ShuffleVectorInstTest.cpp - Shuffle unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ShuffleVectorInst, isIdentityMask) {
  ASSERT_TRUE(ShuffleVectorInst::isIdentityMask({0, 1, 2, 3}, 4));
  ASSERT_TRUE(ShuffleVectorInst::isIdentityMask({0, 1, 2, 3, -1}, 5));
  ASSERT_TRUE(ShuffleVectorInst::isIdentityMask({0, 1, -1, 3}, 4));

  ASSERT_FALSE(ShuffleVectorInst::isIdentityMask({0, 1, 2, 3}, 3));
  ASSERT_FALSE(ShuffleVectorInst::isIdentityMask({0, 1, 2, 3, -1}, 4));
  ASSERT_FALSE(ShuffleVectorInst::isIdentityMask({0, 1, -1, 3}, 3));

  ASSERT_FALSE(ShuffleVectorInst::isIdentityMask({0, 1, 2, 3}, 5));
  ASSERT_FALSE(ShuffleVectorInst::isIdentityMask({0, 1, 2, 3, -1}, 6));
  ASSERT_FALSE(ShuffleVectorInst::isIdentityMask({0, 1, -1, 3}, 5));

  ASSERT_FALSE(ShuffleVectorInst::isIdentityMask({0, 1, 2, 4}, 4));
  ASSERT_FALSE(ShuffleVectorInst::isIdentityMask({0, -1, 2, 4}, 4));
}

TEST(ShuffleVectorInst, isSelectMask) {
  ASSERT_TRUE(ShuffleVectorInst::isSelectMask({0, 5, 6, 3}, 4));

  ASSERT_FALSE(ShuffleVectorInst::isSelectMask({0, 5, 6, 3}, 3));
  ASSERT_FALSE(ShuffleVectorInst::isSelectMask({0, 5, 6, 3}, 5));

  ASSERT_FALSE(ShuffleVectorInst::isSelectMask({0, 1, 2, 3}, 4));
}

TEST(ShuffleVectorInst, isReverseMask) {
  ASSERT_TRUE(ShuffleVectorInst::isReverseMask({3, 2, 1, 0}, 4));
  ASSERT_TRUE(ShuffleVectorInst::isReverseMask({-1, -1, 1, 0}, 4));

  ASSERT_FALSE(ShuffleVectorInst::isReverseMask({3, 2, 1, 0}, 3));
  ASSERT_FALSE(ShuffleVectorInst::isReverseMask({-1, -1, 1, 0}, 3));
  ASSERT_FALSE(ShuffleVectorInst::isReverseMask({3, 2, 1, 0}, 5));
  ASSERT_FALSE(ShuffleVectorInst::isReverseMask({-1, -1, 1, 0}, 5));

  ASSERT_FALSE(ShuffleVectorInst::isReverseMask({4, 3, 2, 1}, 4));
}

TEST(ShuffleVectorInst, isZeroEltSplatMask) {
  ASSERT_TRUE(ShuffleVectorInst::isZeroEltSplatMask({0, 0, 0, 0}, 4));
  ASSERT_TRUE(ShuffleVectorInst::isZeroEltSplatMask({0, -1, 0, -1}, 4));

  ASSERT_FALSE(ShuffleVectorInst::isZeroEltSplatMask({0, 0, 0, 0}, 3));
  ASSERT_FALSE(ShuffleVectorInst::isZeroEltSplatMask({0, -1, 0, -1}, 3));
  ASSERT_FALSE(ShuffleVectorInst::isZeroEltSplatMask({0, 0, 0, 0}, 5));
  ASSERT_FALSE(ShuffleVectorInst::isZeroEltSplatMask({0, -1, 0, -1}, 5));

  ASSERT_FALSE(ShuffleVectorInst::isZeroEltSplatMask({1, 1, 1, 1}, 4));
}

TEST(ShuffleVectorInst, isTransposeMask) {
  ASSERT_TRUE(ShuffleVectorInst::isTransposeMask({0, 4, 2, 6}, 4));
  ASSERT_TRUE(ShuffleVectorInst::isTransposeMask({1, 5, 3, 7}, 4));

  ASSERT_FALSE(ShuffleVectorInst::isTransposeMask({0, 4, 2, 6}, 3));
  ASSERT_FALSE(ShuffleVectorInst::isTransposeMask({1, 5, 3, 7}, 3));
  ASSERT_FALSE(ShuffleVectorInst::isTransposeMask({0, 4, 2, 6}, 5));
  ASSERT_FALSE(ShuffleVectorInst::isTransposeMask({1, 5, 3, 7}, 5));

  ASSERT_FALSE(ShuffleVectorInst::isTransposeMask({2, 6, 4, 8}, 4));
}

TEST(ShuffleVectorInst, isSpliceMask) {
  int Index;

  ASSERT_TRUE(ShuffleVectorInst::isSpliceMask({0, 1, 2, 3}, 4, Index));
  ASSERT_EQ(0, Index);

  ASSERT_TRUE(ShuffleVectorInst::isSpliceMask({1, 2, 3, 4, 5, 6, 7}, 7, Index));
  ASSERT_EQ(1, Index);

  ASSERT_FALSE(ShuffleVectorInst::isSpliceMask({0, 1, 2, 3}, 3, Index));
  ASSERT_FALSE(
      ShuffleVectorInst::isSpliceMask({1, 2, 3, 4, 5, 6, 7}, 6, Index));
  ASSERT_FALSE(ShuffleVectorInst::isSpliceMask({0, 1, 2, 3}, 5, Index));
  ASSERT_FALSE(
      ShuffleVectorInst::isSpliceMask({1, 2, 3, 4, 5, 6, 7}, 8, Index));

  ASSERT_FALSE(ShuffleVectorInst::isSpliceMask({4, 5, 6, 7}, 4, Index));
}

TEST(ShuffleVectorInst, isExtractSubvectorMask) {
  int Index;

  ASSERT_TRUE(
      ShuffleVectorInst::isExtractSubvectorMask({0, 1, 2, 3}, 8, Index));
  ASSERT_EQ(0, Index);

  ASSERT_TRUE(
      ShuffleVectorInst::isExtractSubvectorMask({-1, 3, 4, 5}, 8, Index));
  ASSERT_EQ(2, Index);

  ASSERT_FALSE(
      ShuffleVectorInst::isExtractSubvectorMask({1, 2, 3, -1}, 4, Index));
}

TEST(ShuffleVectorInst, isInsertSubvectorMask) {
  int NumSubElts, Index;

  ASSERT_TRUE(ShuffleVectorInst::isInsertSubvectorMask(
      {8, 9, 10, 11, 4, 5, 6, 7}, 8, NumSubElts, Index));
  ASSERT_EQ(0, Index);
  ASSERT_EQ(4, NumSubElts);

  ASSERT_TRUE(
      ShuffleVectorInst::isInsertSubvectorMask({0, 2}, 2, NumSubElts, Index));
  ASSERT_EQ(1, Index);
  ASSERT_EQ(1, NumSubElts);
}

TEST(ShuffleVectorInst, isReplicationMask) {
  int ReplicationFactor, VF;

  ASSERT_TRUE(ShuffleVectorInst::isReplicationMask({0, 0, 1, 1, 2, 2, 3, 3},
                                                   ReplicationFactor, VF));
  ASSERT_EQ(2, ReplicationFactor);
  ASSERT_EQ(4, VF);

  ASSERT_TRUE(ShuffleVectorInst::isReplicationMask(
      {0, 0, 0, 1, 1, 1, -1, -1, -1, 3, 3, 3, 4, 4, 4}, ReplicationFactor, VF));
  ASSERT_EQ(3, ReplicationFactor);
  ASSERT_EQ(5, VF);
}

TEST(ShuffleVectorInst, isOneUseSingleSourceMask) {
  ASSERT_TRUE(
      ShuffleVectorInst::isOneUseSingleSourceMask({0, 1, 2, 3, 3, 2, 0, 1}, 4));
  ASSERT_TRUE(
      ShuffleVectorInst::isOneUseSingleSourceMask({2, 3, 4, 5, 6, 7, 0, 1}, 8));

  ASSERT_FALSE(ShuffleVectorInst::isOneUseSingleSourceMask(
      {0, -1, 2, 3, 3, 2, 0, 1}, 4));
  ASSERT_FALSE(
      ShuffleVectorInst::isOneUseSingleSourceMask({0, 1, 2, 3, 3, 3, 1, 0}, 4));
}

TEST(ShuffleVectorInst, isInterleaveMask) {
  SmallVector<unsigned> StartIndexes;
  ASSERT_TRUE(ShuffleVectorInst::isInterleaveMask({0, 4, 1, 5, 2, 6, 3, 7}, 2,
                                                  8, StartIndexes));
  ASSERT_EQ(StartIndexes, SmallVector<unsigned>({0, 4}));

  ASSERT_FALSE(
      ShuffleVectorInst::isInterleaveMask({0, 4, 1, 6, 2, 6, 3, 7}, 2, 8));

  ASSERT_TRUE(ShuffleVectorInst::isInterleaveMask({4, 0, 5, 1, 6, 2, 7, 3}, 2,
                                                  8, StartIndexes));
  ASSERT_EQ(StartIndexes, SmallVector<unsigned>({4, 0}));

  ASSERT_TRUE(ShuffleVectorInst::isInterleaveMask({4, 0, -1, 1, -1, 2, 7, 3}, 2,
                                                  8, StartIndexes));
  ASSERT_EQ(StartIndexes, SmallVector<unsigned>({4, 0}));

  ASSERT_TRUE(ShuffleVectorInst::isInterleaveMask({0, 2, 4, 1, 3, 5}, 3, 6,
                                                  StartIndexes));
  ASSERT_EQ(StartIndexes, SmallVector<unsigned>({0, 2, 4}));

  ASSERT_TRUE(ShuffleVectorInst::isInterleaveMask({4, -1, 0, 5, 3, 1}, 3, 6,
                                                  StartIndexes));
  ASSERT_EQ(StartIndexes, SmallVector<unsigned>({4, 2, 0}));

  ASSERT_FALSE(
      ShuffleVectorInst::isInterleaveMask({8, 2, 12, 4, 9, 3, 13, 5}, 4, 8));
}

} // end anonymous namespace
