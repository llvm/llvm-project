//===- llvm/unittest/ADT/BitVectorTest.cpp - BitVector tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Test fixture
template <typename T>
class BitVectorTest : public ::testing::Test { };

// Test both BitVector and SmallBitVector with the same suite of tests.
typedef ::testing::Types<BitVector, SmallBitVector> BitVectorTestTypes;
TYPED_TEST_SUITE(BitVectorTest, BitVectorTestTypes, );

TYPED_TEST(BitVectorTest, TrivialOperation) {
  TypeParam Vec;
  EXPECT_EQ(0U, Vec.count());
  EXPECT_EQ(0U, Vec.size());
  EXPECT_FALSE(Vec.any());
  EXPECT_TRUE(Vec.all());
  EXPECT_TRUE(Vec.none());
  EXPECT_TRUE(Vec.empty());

  Vec.resize(5, true);
  EXPECT_EQ(5U, Vec.count());
  EXPECT_EQ(5U, Vec.size());
  EXPECT_TRUE(Vec.any());
  EXPECT_TRUE(Vec.all());
  EXPECT_FALSE(Vec.none());
  EXPECT_FALSE(Vec.empty());

  Vec.resize(11);
  EXPECT_EQ(5U, Vec.count());
  EXPECT_EQ(11U, Vec.size());
  EXPECT_TRUE(Vec.any());
  EXPECT_FALSE(Vec.all());
  EXPECT_FALSE(Vec.none());
  EXPECT_FALSE(Vec.empty());

  TypeParam Inv = Vec;
  Inv.flip();
  EXPECT_EQ(6U, Inv.count());
  EXPECT_EQ(11U, Inv.size());
  EXPECT_TRUE(Inv.any());
  EXPECT_FALSE(Inv.all());
  EXPECT_FALSE(Inv.none());
  EXPECT_FALSE(Inv.empty());

  EXPECT_FALSE(Inv == Vec);
  EXPECT_TRUE(Inv != Vec);
  Vec.flip();
  EXPECT_TRUE(Inv == Vec);
  EXPECT_FALSE(Inv != Vec);

  // Add some "interesting" data to Vec.
  Vec.resize(23, true);
  Vec.resize(25, false);
  Vec.resize(26, true);
  Vec.resize(29, false);
  Vec.resize(33, true);
  Vec.resize(57, false);
  unsigned Count = 0;
  for (unsigned i = Vec.find_first(); i != -1u; i = Vec.find_next(i)) {
    ++Count;
    EXPECT_TRUE(Vec[i]);
    EXPECT_TRUE(Vec.test(i));
  }
  EXPECT_EQ(Count, Vec.count());
  EXPECT_EQ(Count, 23u);
  EXPECT_FALSE(Vec[0]);
  EXPECT_TRUE(Vec[32]);
  EXPECT_FALSE(Vec[56]);
  Vec.resize(61, false);

  TypeParam Copy = Vec;
  TypeParam Alt(3, false);
  Alt.resize(6, true);
  std::swap(Alt, Vec);
  EXPECT_TRUE(Copy == Alt);
  EXPECT_TRUE(Vec.size() == 6);
  EXPECT_TRUE(Vec.count() == 3);
  EXPECT_TRUE(Vec.find_first() == 3);
  std::swap(Copy, Vec);

  // Add some more "interesting" data.
  Vec.resize(68, true);
  Vec.resize(78, false);
  Vec.resize(89, true);
  Vec.resize(90, false);
  Vec.resize(91, true);
  Vec.resize(130, false);
  Count = 0;
  for (unsigned i = Vec.find_first(); i != -1u; i = Vec.find_next(i)) {
    ++Count;
    EXPECT_TRUE(Vec[i]);
    EXPECT_TRUE(Vec.test(i));
  }
  EXPECT_EQ(Count, Vec.count());
  EXPECT_EQ(Count, 42u);
  EXPECT_FALSE(Vec[0]);
  EXPECT_TRUE(Vec[32]);
  EXPECT_FALSE(Vec[60]);
  EXPECT_FALSE(Vec[129]);

  Vec.flip(60);
  EXPECT_TRUE(Vec[60]);
  EXPECT_EQ(Count + 1, Vec.count());
  Vec.flip(60);
  EXPECT_FALSE(Vec[60]);
  EXPECT_EQ(Count, Vec.count());

  Vec.reset(32);
  EXPECT_FALSE(Vec[32]);
  EXPECT_EQ(Count - 1, Vec.count());
  Vec.set(32);
  EXPECT_TRUE(Vec[32]);
  EXPECT_EQ(Count, Vec.count());

  Vec.flip();
  EXPECT_EQ(Vec.size() - Count, Vec.count());

  Vec.reset();
  EXPECT_EQ(0U, Vec.count());
  EXPECT_EQ(130U, Vec.size());
  EXPECT_FALSE(Vec.any());
  EXPECT_FALSE(Vec.all());
  EXPECT_TRUE(Vec.none());
  EXPECT_FALSE(Vec.empty());

  Vec.flip();
  EXPECT_EQ(130U, Vec.count());
  EXPECT_EQ(130U, Vec.size());
  EXPECT_TRUE(Vec.any());
  EXPECT_TRUE(Vec.all());
  EXPECT_FALSE(Vec.none());
  EXPECT_FALSE(Vec.empty());

  Vec.resize(64);
  EXPECT_EQ(64U, Vec.count());
  EXPECT_EQ(64U, Vec.size());
  EXPECT_TRUE(Vec.any());
  EXPECT_TRUE(Vec.all());
  EXPECT_FALSE(Vec.none());
  EXPECT_FALSE(Vec.empty());

  Vec.flip();
  EXPECT_EQ(0U, Vec.count());
  EXPECT_EQ(64U, Vec.size());
  EXPECT_FALSE(Vec.any());
  EXPECT_FALSE(Vec.all());
  EXPECT_TRUE(Vec.none());
  EXPECT_FALSE(Vec.empty());

  Inv = TypeParam().flip();
  EXPECT_EQ(0U, Inv.count());
  EXPECT_EQ(0U, Inv.size());
  EXPECT_FALSE(Inv.any());
  EXPECT_TRUE(Inv.all());
  EXPECT_TRUE(Inv.none());
  EXPECT_TRUE(Inv.empty());

  Vec.clear();
  EXPECT_EQ(0U, Vec.count());
  EXPECT_EQ(0U, Vec.size());
  EXPECT_FALSE(Vec.any());
  EXPECT_TRUE(Vec.all());
  EXPECT_TRUE(Vec.none());
  EXPECT_TRUE(Vec.empty());
}

TYPED_TEST(BitVectorTest, Equality) {
  TypeParam A;
  TypeParam B;
  EXPECT_TRUE(A == B);
  A.resize(10);
  EXPECT_FALSE(A == B);
  B.resize(10);
  EXPECT_TRUE(A == B);
  A.set(5);
  EXPECT_FALSE(A == B);
  B.set(5);
  EXPECT_TRUE(A == B);
  A.resize(20);
  EXPECT_FALSE(A == B);
  B.resize(20);
  EXPECT_TRUE(A == B);
}

TYPED_TEST(BitVectorTest, SimpleFindOpsMultiWord) {
  TypeParam A;

  // Test finding next set and unset bits in a BitVector with multiple words
  A.resize(100);
  A.set(12);
  A.set(13);
  A.set(75);

  EXPECT_EQ(75, A.find_last());
  EXPECT_EQ(12, A.find_first());
  EXPECT_EQ(13, A.find_next(12));
  EXPECT_EQ(75, A.find_next(13));
  EXPECT_EQ(-1, A.find_next(75));

  EXPECT_EQ(-1, A.find_prev(12));
  EXPECT_EQ(12, A.find_prev(13));
  EXPECT_EQ(13, A.find_prev(75));
  EXPECT_EQ(75, A.find_prev(90));

  EXPECT_EQ(0, A.find_first_unset());
  EXPECT_EQ(99, A.find_last_unset());
  EXPECT_EQ(14, A.find_next_unset(11));
  EXPECT_EQ(14, A.find_next_unset(12));
  EXPECT_EQ(14, A.find_next_unset(13));
  EXPECT_EQ(16, A.find_next_unset(15));
  EXPECT_EQ(76, A.find_next_unset(74));
  EXPECT_EQ(76, A.find_next_unset(75));
  EXPECT_EQ(-1, A.find_next_unset(99));

  A.set(0, 100);
  EXPECT_EQ(100U, A.count());
  EXPECT_EQ(0, A.find_first());
  EXPECT_EQ(-1, A.find_first_unset());
  EXPECT_EQ(-1, A.find_last_unset());
  EXPECT_EQ(99, A.find_last());
  EXPECT_EQ(99, A.find_next(98));

  A.reset(0, 100);
  EXPECT_EQ(0U, A.count());
  EXPECT_EQ(-1, A.find_first());
  EXPECT_EQ(-1, A.find_last());
  EXPECT_EQ(0, A.find_first_unset());
  EXPECT_EQ(99, A.find_last_unset());
  EXPECT_EQ(99, A.find_next_unset(98));
}

// Test finding next set and unset bits in a BitVector/SmallBitVector within a
// uintptr_t - check both 32-bit (Multi) and 64-bit (Small) targets.
TYPED_TEST(BitVectorTest, SimpleFindOps64Bit) {
  TypeParam A;

  A.resize(57);
  A.set(12);
  A.set(13);
  A.set(47);

  EXPECT_EQ(47, A.find_last());
  EXPECT_EQ(12, A.find_first());
  EXPECT_EQ(13, A.find_next(12));
  EXPECT_EQ(47, A.find_next(13));
  EXPECT_EQ(-1, A.find_next(47));

  EXPECT_EQ(-1, A.find_prev(12));
  EXPECT_EQ(12, A.find_prev(13));
  EXPECT_EQ(13, A.find_prev(47));
  EXPECT_EQ(47, A.find_prev(56));

  EXPECT_EQ(0, A.find_first_unset());
  EXPECT_EQ(56, A.find_last_unset());
  EXPECT_EQ(14, A.find_next_unset(11));
  EXPECT_EQ(14, A.find_next_unset(12));
  EXPECT_EQ(14, A.find_next_unset(13));
  EXPECT_EQ(16, A.find_next_unset(15));
  EXPECT_EQ(48, A.find_next_unset(46));
  EXPECT_EQ(48, A.find_next_unset(47));
  EXPECT_EQ(-1, A.find_next_unset(56));
}

// Check if a SmallBitVector is in small mode. This check is used in tests
// that run for both SmallBitVector and BitVector. This check doesn't apply
// to BitVector so we provide an overload that returns true to get the tests
// to compile.
static bool SmallBitVectorIsSmallMode(const SmallBitVector &bv) {
  return bv.isSmall();
}
static bool SmallBitVectorIsSmallMode(const BitVector &) { return true; }

// These tests are intended to exercise the single-word case of BitVector
// and the small-mode case of SmallBitVector.
TYPED_TEST(BitVectorTest, SimpleFindOpsSingleWord) {
  // Test finding in an empty BitVector.
  TypeParam A;
  ASSERT_TRUE(SmallBitVectorIsSmallMode(A));
  EXPECT_EQ(-1, A.find_first());
  EXPECT_EQ(-1, A.find_last());
  EXPECT_EQ(-1, A.find_first_unset());
  EXPECT_EQ(-1, A.find_last_unset());

  A.resize(20);
  ASSERT_TRUE(SmallBitVectorIsSmallMode(A));
  EXPECT_EQ(-1, A.find_first());
  EXPECT_EQ(-1, A.find_last());
  EXPECT_EQ(-1, A.find_next(5));
  EXPECT_EQ(-1, A.find_next(19));
  EXPECT_EQ(-1, A.find_prev(5));
  EXPECT_EQ(-1, A.find_prev(20));
  EXPECT_EQ(0, A.find_first_unset());
  EXPECT_EQ(19, A.find_last_unset());
  EXPECT_EQ(6, A.find_next_unset(5));
  EXPECT_EQ(-1, A.find_next_unset(19));

  A.set(3);
  A.set(4);
  A.set(16);
  ASSERT_TRUE(SmallBitVectorIsSmallMode(A));
  EXPECT_EQ(16, A.find_last());
  EXPECT_EQ(3, A.find_first());
  EXPECT_EQ(3, A.find_next(1));
  EXPECT_EQ(4, A.find_next(3));
  EXPECT_EQ(16, A.find_next(4));
  EXPECT_EQ(-1, A.find_next(16));

  EXPECT_EQ(-1, A.find_prev(3));
  EXPECT_EQ(3, A.find_prev(4));
  EXPECT_EQ(4, A.find_prev(16));
  EXPECT_EQ(16, A.find_prev(18));

  EXPECT_EQ(0, A.find_first_unset());
  EXPECT_EQ(19, A.find_last_unset());
  EXPECT_EQ(5, A.find_next_unset(3));
  EXPECT_EQ(5, A.find_next_unset(4));
  EXPECT_EQ(13, A.find_next_unset(12));
  EXPECT_EQ(17, A.find_next_unset(15));

  A.set();
  ASSERT_TRUE(SmallBitVectorIsSmallMode(A));
  EXPECT_EQ(0, A.find_first());
  EXPECT_EQ(19, A.find_last());
  EXPECT_EQ(6, A.find_next(5));
  EXPECT_EQ(-1, A.find_next(19));
  EXPECT_EQ(4, A.find_prev(5));
  EXPECT_EQ(19, A.find_prev(20));
  EXPECT_EQ(-1, A.find_first_unset());
  EXPECT_EQ(-1, A.find_last_unset());
  EXPECT_EQ(-1, A.find_next_unset(5));
  EXPECT_EQ(-1, A.find_next_unset(19));
}

TEST(BitVectorTest, FindInRangeMultiWord) {
  BitVector Vec;

  Vec.resize(200);
  Vec.set(3, 7);
  Vec.set(24, 35);
  Vec.set(50, 70);
  Vec.set(150);
  Vec.set(152);
  Vec.set(154);

  // find first
  EXPECT_EQ(-1, Vec.find_first_in(0, 0));
  EXPECT_EQ(-1, Vec.find_first_in(24, 24));
  EXPECT_EQ(-1, Vec.find_first_in(7, 24));

  EXPECT_EQ(3, Vec.find_first_in(0, 10));
  EXPECT_EQ(4, Vec.find_first_in(4, 10));
  EXPECT_EQ(150, Vec.find_first_in(100, 200));
  EXPECT_EQ(152, Vec.find_first_in(151, 200));
  EXPECT_EQ(154, Vec.find_first_in(153, 200));

  EXPECT_EQ(-1, Vec.find_first_in(155, 200));
  Vec.set(199);
  EXPECT_EQ(199, Vec.find_first_in(199, 200));
  Vec.reset(199);

  // find last
  EXPECT_EQ(-1, Vec.find_last_in(0, 0));
  EXPECT_EQ(-1, Vec.find_last_in(24, 24));
  EXPECT_EQ(-1, Vec.find_last_in(7, 24));

  EXPECT_EQ(6, Vec.find_last_in(0, 10));
  EXPECT_EQ(5, Vec.find_last_in(0, 6));
  EXPECT_EQ(154, Vec.find_last_in(100, 155));
  EXPECT_EQ(152, Vec.find_last_in(100, 154));
  EXPECT_EQ(150, Vec.find_last_in(100, 152));
  EXPECT_EQ(-1, Vec.find_last_in(100, 150));
  Vec.set(199);
  EXPECT_EQ(199, Vec.find_last_in(199, 200));
  Vec.reset(199);

  // find first unset
  EXPECT_EQ(-1, Vec.find_first_unset_in(0, 0));
  EXPECT_EQ(-1, Vec.find_first_unset_in(23, 23));
  EXPECT_EQ(-1, Vec.find_first_unset_in(24, 35));

  EXPECT_EQ(0, Vec.find_first_unset_in(0, 10));
  EXPECT_EQ(1, Vec.find_first_unset_in(1, 10));
  EXPECT_EQ(7, Vec.find_first_unset_in(5, 25));
  EXPECT_EQ(151, Vec.find_first_unset_in(150, 200));
  EXPECT_EQ(151, Vec.find_first_unset_in(151, 200));
  EXPECT_EQ(153, Vec.find_first_unset_in(152, 200));
  EXPECT_EQ(153, Vec.find_first_unset_in(153, 200));
  EXPECT_EQ(155, Vec.find_first_unset_in(154, 200));
  EXPECT_EQ(155, Vec.find_first_unset_in(155, 200));
  EXPECT_EQ(199, Vec.find_first_unset_in(199, 200));

  // find last unset
  EXPECT_EQ(-1, Vec.find_last_unset_in(0, 0));
  EXPECT_EQ(-1, Vec.find_last_unset_in(23, 23));
  EXPECT_EQ(-1, Vec.find_last_unset_in(24, 35));

  EXPECT_EQ(9, Vec.find_last_unset_in(0, 10));
  EXPECT_EQ(8, Vec.find_last_unset_in(0, 9));
  EXPECT_EQ(2, Vec.find_last_unset_in(0, 7));
  EXPECT_EQ(149, Vec.find_last_unset_in(100, 151));
  EXPECT_EQ(151, Vec.find_last_unset_in(100, 152));
  EXPECT_EQ(151, Vec.find_last_unset_in(100, 153));
  EXPECT_EQ(153, Vec.find_last_unset_in(100, 154));
  EXPECT_EQ(153, Vec.find_last_unset_in(100, 155));
  EXPECT_EQ(155, Vec.find_last_unset_in(100, 156));
  EXPECT_EQ(199, Vec.find_last_unset_in(199, 200));
}

TEST(BitVectorTest, FindInRangeSingleWord) {
  // When the bit vector contains only a single word, this is slightly different
  // than when the bit vector contains multiple words, because masks are applied
  // to the front and back of the same word.  So make sure this works.
  BitVector Vec;

  Vec.resize(25);
  Vec.set(2, 4);
  Vec.set(6, 9);
  Vec.set(12, 15);
  Vec.set(19);
  Vec.set(21);
  Vec.set(23);

  // find first
  EXPECT_EQ(-1, Vec.find_first_in(0, 0));
  EXPECT_EQ(-1, Vec.find_first_in(24, 24));
  EXPECT_EQ(-1, Vec.find_first_in(9, 12));

  EXPECT_EQ(2, Vec.find_first_in(0, 10));
  EXPECT_EQ(6, Vec.find_first_in(4, 10));
  EXPECT_EQ(19, Vec.find_first_in(18, 25));
  EXPECT_EQ(21, Vec.find_first_in(20, 25));
  EXPECT_EQ(23, Vec.find_first_in(22, 25));
  EXPECT_EQ(-1, Vec.find_first_in(24, 25));

  // find last
  EXPECT_EQ(-1, Vec.find_last_in(0, 0));
  EXPECT_EQ(-1, Vec.find_last_in(24, 24));
  EXPECT_EQ(-1, Vec.find_last_in(9, 12));

  EXPECT_EQ(8, Vec.find_last_in(0, 10));
  EXPECT_EQ(3, Vec.find_last_in(0, 6));
  EXPECT_EQ(23, Vec.find_last_in(18, 25));
  EXPECT_EQ(21, Vec.find_last_in(18, 23));
  EXPECT_EQ(19, Vec.find_last_in(18, 21));
  EXPECT_EQ(-1, Vec.find_last_in(18, 19));

  // find first unset
  EXPECT_EQ(-1, Vec.find_first_unset_in(0, 0));
  EXPECT_EQ(-1, Vec.find_first_unset_in(23, 23));
  EXPECT_EQ(-1, Vec.find_first_unset_in(6, 9));

  EXPECT_EQ(0, Vec.find_first_unset_in(0, 6));
  EXPECT_EQ(1, Vec.find_first_unset_in(1, 6));
  EXPECT_EQ(9, Vec.find_first_unset_in(7, 13));
  EXPECT_EQ(18, Vec.find_first_unset_in(18, 25));
  EXPECT_EQ(20, Vec.find_first_unset_in(19, 25));
  EXPECT_EQ(20, Vec.find_first_unset_in(20, 25));
  EXPECT_EQ(22, Vec.find_first_unset_in(21, 25));
  EXPECT_EQ(22, Vec.find_first_unset_in(22, 25));
  EXPECT_EQ(24, Vec.find_first_unset_in(23, 25));
  EXPECT_EQ(24, Vec.find_first_unset_in(24, 25));

  // find last unset
  EXPECT_EQ(-1, Vec.find_last_unset_in(0, 0));
  EXPECT_EQ(-1, Vec.find_last_unset_in(23, 23));
  EXPECT_EQ(-1, Vec.find_last_unset_in(6, 9));

  EXPECT_EQ(5, Vec.find_last_unset_in(0, 6));
  EXPECT_EQ(4, Vec.find_last_unset_in(0, 5));
  EXPECT_EQ(1, Vec.find_last_unset_in(0, 4));
  EXPECT_EQ(11, Vec.find_last_unset_in(7, 13));
  EXPECT_EQ(24, Vec.find_last_unset_in(18, 25));
  EXPECT_EQ(22, Vec.find_last_unset_in(18, 24));
  EXPECT_EQ(22, Vec.find_last_unset_in(18, 23));
  EXPECT_EQ(20, Vec.find_last_unset_in(18, 22));
  EXPECT_EQ(20, Vec.find_last_unset_in(18, 21));
  EXPECT_EQ(18, Vec.find_last_unset_in(18, 20));
  EXPECT_EQ(18, Vec.find_last_unset_in(18, 19));
}

TYPED_TEST(BitVectorTest, CompoundAssignment) {
  TypeParam A;
  A.resize(10);
  A.set(4);
  A.set(7);

  TypeParam B;
  B.resize(50);
  B.set(5);
  B.set(18);

  A |= B;
  EXPECT_TRUE(A.test(4));
  EXPECT_TRUE(A.test(5));
  EXPECT_TRUE(A.test(7));
  EXPECT_TRUE(A.test(18));
  EXPECT_EQ(4U, A.count());
  EXPECT_EQ(50U, A.size());

  B.resize(10);
  B.set();
  B.reset(2);
  B.reset(7);
  A &= B;
  EXPECT_FALSE(A.test(2));
  EXPECT_FALSE(A.test(7));
  EXPECT_TRUE(A.test(4));
  EXPECT_TRUE(A.test(5));
  EXPECT_EQ(2U, A.count());
  EXPECT_EQ(50U, A.size());

  B.resize(100);
  B.set();

  A ^= B;
  EXPECT_TRUE(A.test(2));
  EXPECT_TRUE(A.test(7));
  EXPECT_EQ(98U, A.count());
  EXPECT_EQ(100U, A.size());
}

// Test SmallBitVector operations with mixed big/small representations
TYPED_TEST(BitVectorTest, MixedBigSmall) {
  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(20);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);
    Big.set(2);
    Big.set(16);

    Small &= Big;
    EXPECT_TRUE(Small.test(0));
    EXPECT_EQ(1u, Small.count());
    // FIXME BitVector and SmallBitVector behave differently here.
    // SmallBitVector resizes the LHS to max(LHS.size(), RHS.size())
    // but BitVector does not.
    // EXPECT_EQ(20u, Small.size());
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(20);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);
    Big.set(2);
    Big.set(16);

    Big &= Small;
    EXPECT_TRUE(Big.test(0));
    EXPECT_EQ(1u, Big.count());
    // FIXME BitVector and SmallBitVector behave differently here.
    // SmallBitVector resizes the LHS to max(LHS.size(), RHS.size())
    // but BitVector does not.
    // EXPECT_EQ(20u, Big.size());
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(20);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);
    Big.set(2);
    Big.set(16);

    Small |= Big;
    EXPECT_TRUE(Small.test(0));
    EXPECT_TRUE(Small.test(1));
    EXPECT_TRUE(Small.test(2));
    EXPECT_TRUE(Small.test(16));
    EXPECT_EQ(4u, Small.count());
    EXPECT_EQ(20u, Small.size());
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(20);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);
    Big.set(2);
    Big.set(16);

    Big |= Small;
    EXPECT_TRUE(Big.test(0));
    EXPECT_TRUE(Big.test(1));
    EXPECT_TRUE(Big.test(2));
    EXPECT_TRUE(Big.test(16));
    EXPECT_EQ(4u, Big.count());
    EXPECT_EQ(20u, Big.size());
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(20);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);
    Big.set(2);
    Big.set(16);

    Small ^= Big;
    EXPECT_TRUE(Small.test(1));
    EXPECT_TRUE(Small.test(2));
    EXPECT_TRUE(Small.test(16));
    EXPECT_EQ(3u, Small.count());
    EXPECT_EQ(20u, Small.size());
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(20);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);
    Big.set(2);
    Big.set(16);

    Big ^= Small;
    EXPECT_TRUE(Big.test(1));
    EXPECT_TRUE(Big.test(2));
    EXPECT_TRUE(Big.test(16));
    EXPECT_EQ(3u, Big.count());
    EXPECT_EQ(20u, Big.size());
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(20);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);
    Big.set(2);
    Big.set(16);

    Small.reset(Big);
    EXPECT_TRUE(Small.test(1));
    EXPECT_EQ(1u, Small.count());
    EXPECT_EQ(10u, Small.size());
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(20);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);
    Big.set(2);
    Big.set(16);

    Big.reset(Small);
    EXPECT_TRUE(Big.test(2));
    EXPECT_TRUE(Big.test(16));
    EXPECT_EQ(2u, Big.count());
    EXPECT_EQ(20u, Big.size());
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(10);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);

    EXPECT_FALSE(Big == Small);
    EXPECT_FALSE(Small == Big);
    Big.set(1);
    EXPECT_TRUE(Big == Small);
    EXPECT_TRUE(Small == Big);
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(20);
    Small.resize(10);

    Small.set(0);
    Big.set(1);

    EXPECT_FALSE(Small.anyCommon(Big));
    EXPECT_FALSE(Big.anyCommon(Small));
    Big.set(0);
    EXPECT_TRUE(Small.anyCommon(Big));
    EXPECT_TRUE(Big.anyCommon(Small));
  }

  {
    TypeParam Big;
    TypeParam Small;

    Big.reserve(100);
    Big.resize(10);
    Small.resize(10);

    Small.set(0);
    Small.set(1);
    Big.set(0);

    EXPECT_TRUE(Small.test(Big));
    EXPECT_FALSE(Big.test(Small));
    Big.set(1);
    EXPECT_FALSE(Small.test(Big));
    EXPECT_FALSE(Big.test(Small));
  }
}

TYPED_TEST(BitVectorTest, ProxyIndex) {
  TypeParam Vec(3);
  EXPECT_TRUE(Vec.none());
  Vec[0] = Vec[1] = Vec[2] = true;
  EXPECT_EQ(Vec.size(), Vec.count());
  Vec[2] = Vec[1] = Vec[0] = false;
  EXPECT_TRUE(Vec.none());
}


TYPED_TEST(BitVectorTest, PortableBitMask) {
  TypeParam A;
  const uint32_t Mask1[] = { 0x80000000, 6, 5 };

  A.resize(10);
  A.setBitsInMask(Mask1, 1);
  EXPECT_EQ(10u, A.size());
  EXPECT_FALSE(A.test(0));

  A.resize(32);
  A.setBitsInMask(Mask1, 1);
  EXPECT_FALSE(A.test(0));
  EXPECT_TRUE(A.test(31));
  EXPECT_EQ(1u, A.count());

  A.resize(33);
  A.setBitsInMask(Mask1, 1);
  EXPECT_EQ(1u, A.count());
  A.setBitsInMask(Mask1, 2);
  EXPECT_EQ(1u, A.count());

  A.resize(34);
  A.setBitsInMask(Mask1, 2);
  EXPECT_EQ(2u, A.count());

  A.resize(65);
  A.setBitsInMask(Mask1, 3);
  EXPECT_EQ(4u, A.count());

  A.setBitsNotInMask(Mask1, 1);
  EXPECT_EQ(32u+3u, A.count());

  A.setBitsNotInMask(Mask1, 3);
  EXPECT_EQ(65u, A.count());

  A.resize(96);
  EXPECT_EQ(65u, A.count());

  A.clear();
  A.resize(128);
  A.setBitsNotInMask(Mask1, 3);
  EXPECT_EQ(96u-5u, A.count());

  A.clearBitsNotInMask(Mask1, 1);
  EXPECT_EQ(64-4u, A.count());
}

TYPED_TEST(BitVectorTest, BinOps) {
  TypeParam A;
  TypeParam B;

  A.resize(65);
  EXPECT_FALSE(A.anyCommon(B));
  EXPECT_FALSE(B.anyCommon(B));

  B.resize(64);
  A.set(64);
  EXPECT_FALSE(A.anyCommon(B));
  EXPECT_FALSE(B.anyCommon(A));

  B.set(63);
  EXPECT_FALSE(A.anyCommon(B));
  EXPECT_FALSE(B.anyCommon(A));

  A.set(63);
  EXPECT_TRUE(A.anyCommon(B));
  EXPECT_TRUE(B.anyCommon(A));

  B.resize(70);
  B.set(64);
  B.reset(63);
  A.resize(64);
  EXPECT_FALSE(A.anyCommon(B));
  EXPECT_FALSE(B.anyCommon(A));
}

typedef std::vector<std::pair<int, int>> RangeList;

template <typename VecType>
static inline VecType createBitVector(uint32_t Size,
                                      const RangeList &setRanges) {
  VecType V;
  V.resize(Size);
  for (auto &R : setRanges)
    V.set(R.first, R.second);
  return V;
}

TYPED_TEST(BitVectorTest, ShiftOpsSingleWord) {
  // Test that shift ops work when the desired shift amount is less
  // than one word.

  // 1. Case where the number of bits in the BitVector also fit into a single
  // word.
  TypeParam A = createBitVector<TypeParam>(12, {{2, 4}, {8, 10}});
  TypeParam B = A;

  EXPECT_EQ(4U, A.count());
  EXPECT_TRUE(A.test(2));
  EXPECT_TRUE(A.test(3));
  EXPECT_TRUE(A.test(8));
  EXPECT_TRUE(A.test(9));

  A >>= 1;
  EXPECT_EQ(createBitVector<TypeParam>(12, {{1, 3}, {7, 9}}), A);

  A <<= 1;
  EXPECT_EQ(B, A);

  A >>= 10;
  EXPECT_EQ(createBitVector<TypeParam>(12, {}), A);

  A = B;
  A <<= 10;
  EXPECT_EQ(createBitVector<TypeParam>(12, {}), A);

  // 2. Case where the number of bits in the BitVector do not fit into a single
  // word.

  // 31----------------------------------------------------------------------0
  // XXXXXXXX XXXXXXXX XXXXXXXX 00000111 | 11111110 00000000 00001111 11111111
  A = createBitVector<TypeParam>(40, {{0, 12}, {25, 35}});
  EXPECT_EQ(40U, A.size());
  EXPECT_EQ(22U, A.count());

  // 2a. Make sure that left shifting some 1 bits out of the vector works.
  //   31----------------------------------------------------------------------0
  // Before:
  //   XXXXXXXX XXXXXXXX XXXXXXXX 00000111 | 11111110 00000000 00001111 11111111
  // After:
  //   XXXXXXXX XXXXXXXX XXXXXXXX 11111100 | 00000000 00011111 11111110 00000000
  A <<= 9;
  EXPECT_EQ(createBitVector<TypeParam>(40, {{9, 21}, {34, 40}}), A);

  // 2b. Make sure that keeping the number of one bits unchanged works.
  //   31----------------------------------------------------------------------0
  // Before:
  //   XXXXXXXX XXXXXXXX XXXXXXXX 11111100 | 00000000 00011111 11111110 00000000
  // After:
  //   XXXXXXXX XXXXXXXX XXXXXXXX 00000011 | 11110000 00000000 01111111 11111000
  A >>= 6;
  EXPECT_EQ(createBitVector<TypeParam>(40, {{3, 15}, {28, 34}}), A);

  // 2c. Make sure that right shifting some 1 bits out of the vector works.
  //   31----------------------------------------------------------------------0
  // Before:
  //   XXXXXXXX XXXXXXXX XXXXXXXX 00000011 | 11110000 00000000 01111111 11111000
  // After:
  //   XXXXXXXX XXXXXXXX XXXXXXXX 00000000 | 00000000 11111100 00000000 00011111
  A >>= 10;
  EXPECT_EQ(createBitVector<TypeParam>(40, {{0, 5}, {18, 24}}), A);

  // 3. Big test.
  A = createBitVector<TypeParam>(300, {{1, 30}, {60, 95}, {200, 275}});
  A <<= 29;
  EXPECT_EQ(createBitVector<TypeParam>(
                300, {{1 + 29, 30 + 29}, {60 + 29, 95 + 29}, {200 + 29, 300}}),
            A);
}

TYPED_TEST(BitVectorTest, ShiftOpsMultiWord) {
  // Test that shift ops work when the desired shift amount is greater than or
  // equal to the size of a single word.
  auto A = createBitVector<TypeParam>(300, {{1, 30}, {60, 95}, {200, 275}});

  // Make a copy so we can re-use it later.
  auto B = A;

  // 1. Shift left by an exact multiple of the word size.  This should invoke
  // only a memmove and no per-word bit operations.
  A <<= 64;
  auto Expected = createBitVector<TypeParam>(
      300, {{1 + 64, 30 + 64}, {60 + 64, 95 + 64}, {200 + 64, 300}});
  EXPECT_EQ(Expected, A);

  // 2. Shift left by a non multiple of the word size.  This should invoke both
  // a memmove and per-word bit operations.
  A = B;
  A <<= 93;
  EXPECT_EQ(createBitVector<TypeParam>(
                300, {{1 + 93, 30 + 93}, {60 + 93, 95 + 93}, {200 + 93, 300}}),
            A);

  // 1. Shift right by an exact multiple of the word size.  This should invoke
  // only a memmove and no per-word bit operations.
  A = B;
  A >>= 64;
  EXPECT_EQ(
      createBitVector<TypeParam>(300, {{0, 95 - 64}, {200 - 64, 275 - 64}}), A);

  // 2. Shift left by a non multiple of the word size.  This should invoke both
  // a memmove and per-word bit operations.
  A = B;
  A >>= 93;
  EXPECT_EQ(
      createBitVector<TypeParam>(300, {{0, 95 - 93}, {200 - 93, 275 - 93}}), A);
}

TYPED_TEST(BitVectorTest, RangeOps) {
  TypeParam A;
  A.resize(256);
  A.reset();
  A.set(1, 255);

  EXPECT_FALSE(A.test(0));
  EXPECT_TRUE( A.test(1));
  EXPECT_TRUE( A.test(23));
  EXPECT_TRUE( A.test(254));
  EXPECT_FALSE(A.test(255));

  TypeParam B;
  B.resize(256);
  B.set();
  B.reset(1, 255);

  EXPECT_TRUE( B.test(0));
  EXPECT_FALSE(B.test(1));
  EXPECT_FALSE(B.test(23));
  EXPECT_FALSE(B.test(254));
  EXPECT_TRUE( B.test(255));

  TypeParam C;
  C.resize(3);
  C.reset();
  C.set(0, 1);

  EXPECT_TRUE(C.test(0));
  EXPECT_FALSE( C.test(1));
  EXPECT_FALSE( C.test(2));

  TypeParam D;
  D.resize(3);
  D.set();
  D.reset(0, 1);

  EXPECT_FALSE(D.test(0));
  EXPECT_TRUE( D.test(1));
  EXPECT_TRUE( D.test(2));

  TypeParam E;
  E.resize(128);
  E.reset();
  E.set(1, 33);

  EXPECT_FALSE(E.test(0));
  EXPECT_TRUE( E.test(1));
  EXPECT_TRUE( E.test(32));
  EXPECT_FALSE(E.test(33));

  TypeParam BufferOverrun;
  unsigned size = sizeof(unsigned long) * 8;
  BufferOverrun.resize(size);
  BufferOverrun.reset(0, size);
  BufferOverrun.set(0, size);
}

TYPED_TEST(BitVectorTest, CompoundTestReset) {
  TypeParam A(50, true);
  TypeParam B(50, false);

  TypeParam C(100, true);
  TypeParam D(100, false);

  EXPECT_FALSE(A.test(A));
  EXPECT_TRUE(A.test(B));
  EXPECT_FALSE(A.test(C));
  EXPECT_TRUE(A.test(D));
  EXPECT_FALSE(B.test(A));
  EXPECT_FALSE(B.test(B));
  EXPECT_FALSE(B.test(C));
  EXPECT_FALSE(B.test(D));
  EXPECT_TRUE(C.test(A));
  EXPECT_TRUE(C.test(B));
  EXPECT_FALSE(C.test(C));
  EXPECT_TRUE(C.test(D));

  A.reset(B);
  A.reset(D);
  EXPECT_TRUE(A.all());
  A.reset(A);
  EXPECT_TRUE(A.none());
  A.set();
  A.reset(C);
  EXPECT_TRUE(A.none());
  A.set();

  C.reset(A);
  EXPECT_EQ(50, C.find_first());
  C.reset(C);
  EXPECT_TRUE(C.none());
}

TYPED_TEST(BitVectorTest, MoveConstructor) {
  TypeParam A(10, true);
  TypeParam B(std::move(A));
  // Check that the move ctor leaves the moved-from object in a valid state.
  // The following line used to crash.
  A = B;

  TypeParam C(10, true);
  EXPECT_EQ(C, A);
  EXPECT_EQ(C, B);
}

TYPED_TEST(BitVectorTest, MoveAssignment) {
  TypeParam A(10, true);
  TypeParam B;
  B = std::move(A);
  // Check that move assignment leaves the moved-from object in a valid state.
  // The following line used to crash.
  A = B;

  TypeParam C(10, true);
  EXPECT_EQ(C, A);
  EXPECT_EQ(C, B);
}

template<class TypeParam>
static void testEmpty(const TypeParam &A) {
  EXPECT_TRUE(A.empty());
  EXPECT_EQ((size_t)0, A.size());
  EXPECT_EQ((size_t)0, A.count());
  EXPECT_FALSE(A.any());
  EXPECT_TRUE(A.all());
  EXPECT_TRUE(A.none());
  EXPECT_EQ(-1, A.find_first());
  EXPECT_EQ(A, TypeParam());
}

/// Tests whether BitVector behaves well with Bits==nullptr, Capacity==0
TYPED_TEST(BitVectorTest, EmptyVector) {
  TypeParam A;
  testEmpty(A);

  TypeParam B;
  B.reset();
  testEmpty(B);

  TypeParam C;
  C.clear();
  testEmpty(C);

  TypeParam D(A);
  testEmpty(D);

  TypeParam E;
  E = A;
  testEmpty(E);

  TypeParam F;
  E.reset(A);
  testEmpty(E);
}

TYPED_TEST(BitVectorTest, Iterators) {
  TypeParam Filled(10, true);
  EXPECT_NE(Filled.set_bits_begin(), Filled.set_bits_end());
  unsigned Counter = 0;
  for (unsigned Bit : Filled.set_bits())
    EXPECT_EQ(Bit, Counter++);

  TypeParam Empty;
  EXPECT_EQ(Empty.set_bits_begin(), Empty.set_bits_end());
  int BitCount = 0;
  for (unsigned Bit : Empty.set_bits()) {
    (void)Bit;
    BitCount++;
  }
  ASSERT_EQ(BitCount, 0);

  TypeParam ToFill(100, false);
  ToFill.set(0);
  EXPECT_NE(ToFill.set_bits_begin(), ToFill.set_bits_end());
  EXPECT_EQ(++ToFill.set_bits_begin(), ToFill.set_bits_end());
  EXPECT_EQ(*ToFill.set_bits_begin(), 0U);
  ToFill.reset(0);
  EXPECT_EQ(ToFill.set_bits_begin(), ToFill.set_bits_end());

  const unsigned List[] = {1, 10, 25, 99};
  for (unsigned Num : List)
    ToFill.set(Num);
  unsigned i = 0;
  for (unsigned Bit : ToFill.set_bits())
    EXPECT_EQ(List[i++], Bit);
}

TYPED_TEST(BitVectorTest, PushBack) {
  TypeParam Vec(10, false);
  EXPECT_EQ(-1, Vec.find_first());
  EXPECT_EQ(10U, Vec.size());
  EXPECT_EQ(0U, Vec.count());
  EXPECT_EQ(false, Vec.back());

  Vec.push_back(true);
  EXPECT_EQ(10, Vec.find_first());
  EXPECT_EQ(11U, Vec.size());
  EXPECT_EQ(1U, Vec.count());
  EXPECT_EQ(true, Vec.back());

  Vec.push_back(false);
  EXPECT_EQ(10, Vec.find_first());
  EXPECT_EQ(12U, Vec.size());
  EXPECT_EQ(1U, Vec.count());
  EXPECT_EQ(false, Vec.back());

  Vec.push_back(true);
  EXPECT_EQ(10, Vec.find_first());
  EXPECT_EQ(13U, Vec.size());
  EXPECT_EQ(2U, Vec.count());
  EXPECT_EQ(true, Vec.back());

  // Add a lot of values to cause reallocation.
  for (int i = 0; i != 100; ++i) {
    Vec.push_back(true);
    Vec.push_back(false);
  }
  EXPECT_EQ(10, Vec.find_first());
  EXPECT_EQ(213U, Vec.size());
  EXPECT_EQ(102U, Vec.count());
}

TYPED_TEST(BitVectorTest, PopBack) {
  TypeParam Vec(10, true);
  EXPECT_EQ(10U, Vec.size());
  EXPECT_EQ(10U, Vec.count());
  EXPECT_EQ(true, Vec.back());

  Vec.pop_back();
  EXPECT_EQ(9U, Vec.size());
  EXPECT_EQ(9U, Vec.count());
  EXPECT_EQ(true, Vec.back());

  Vec.push_back(false);
  EXPECT_EQ(10U, Vec.size());
  EXPECT_EQ(9U, Vec.count());
  EXPECT_EQ(false, Vec.back());

  Vec.pop_back();
  EXPECT_EQ(9U, Vec.size());
  EXPECT_EQ(9U, Vec.count());
  EXPECT_EQ(true, Vec.back());
}

TYPED_TEST(BitVectorTest, DenseSet) {
  DenseSet<TypeParam> Set;
  TypeParam A(10, true);
  auto I = Set.insert(A);
  EXPECT_EQ(true, I.second);

  TypeParam B(5, true);
  I = Set.insert(B);
  EXPECT_EQ(true, I.second);

  TypeParam C(20, false);
  C.set(19);
  I = Set.insert(C);
  EXPECT_EQ(true, I.second);

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  TypeParam D;
  EXPECT_DEATH(Set.insert(D),
               "Empty/Tombstone value shouldn't be inserted into map!");
#endif

  EXPECT_EQ(3U, Set.size());
  EXPECT_EQ(1U, Set.count(A));
  EXPECT_EQ(1U, Set.count(B));
  EXPECT_EQ(1U, Set.count(C));

  EXPECT_EQ(true, Set.erase(B));
  EXPECT_EQ(2U, Set.size());

  EXPECT_EQ(true, Set.erase(C));
  EXPECT_EQ(1U, Set.size());

  EXPECT_EQ(true, Set.erase(A));
  EXPECT_EQ(0U, Set.size());
}

/// Test that capacity doesn't affect hashing.
TYPED_TEST(BitVectorTest, DenseMapHashing) {
  using DMI = DenseMapInfo<TypeParam>;
  {
    TypeParam A;
    A.resize(200);
    A.set(100);

    TypeParam B;
    B.resize(200);
    B.set(100);
    B.reserve(1000);

    EXPECT_EQ(DMI::getHashValue(A), DMI::getHashValue(B));
  }
  {
    TypeParam A;
    A.resize(20);
    A.set(10);

    TypeParam B;
    B.resize(20);
    B.set(10);
    B.reserve(1000);

    EXPECT_EQ(DMI::getHashValue(A), DMI::getHashValue(B));
  }
}

TEST(BitVectoryTest, Apply) {
  for (int i = 0; i < 2; ++i) {
    int j = i * 100 + 3;

    const BitVector x =
        createBitVector<BitVector>(j + 5, {{i, i + 1}, {j - 1, j}});
    const BitVector y = createBitVector<BitVector>(j + 5, {{i, j}});
    const BitVector z =
        createBitVector<BitVector>(j + 5, {{i + 1, i + 2}, {j, j + 1}});

    auto op0 = [](auto x) { return ~x; };
    BitVector expected0 = x;
    expected0.flip();
    BitVector out0(j - 2);
    BitVector::apply(op0, out0, x);
    EXPECT_EQ(out0, expected0);

    auto op1 = [](auto x, auto y) { return x & ~y; };
    BitVector expected1 = x;
    expected1.reset(y);
    BitVector out1;
    BitVector::apply(op1, out1, x, y);
    EXPECT_EQ(out1, expected1);

    auto op2 = [](auto x, auto y, auto z) { return (x ^ ~y) | z; };
    BitVector expected2 = y;
    expected2.flip();
    expected2 ^= x;
    expected2 |= z;
    BitVector out2(j + 5);
    BitVector::apply(op2, out2, x, y, z);
    EXPECT_EQ(out2, expected2);
  }
}


} // namespace
