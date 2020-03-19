//=== CoalescingBitVectorTest.cpp - CoalescingBitVector unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/CoalescingBitVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

using UBitVec = CoalescingBitVector<unsigned>;
using U64BitVec = CoalescingBitVector<uint64_t>;

bool elementsMatch(const UBitVec &BV, std::initializer_list<unsigned> List) {
  if (!std::equal(BV.begin(), BV.end(), List.begin(), List.end())) {
    UBitVec::Allocator Alloc;
    UBitVec Expected(Alloc);
    Expected.set(List);
    dbgs() << "elementsMatch:\n"
           << "     Expected: ";
    Expected.print(dbgs());
    dbgs() << "          Got: ";
    BV.print(dbgs());
    return false;
  }
  return true;
}

TEST(CoalescingBitVectorTest, Set) {
  UBitVec::Allocator Alloc;
  UBitVec BV1(Alloc);
  UBitVec BV2(Alloc);

  BV1.set(0);
  EXPECT_TRUE(BV1.test(0));
  EXPECT_FALSE(BV1.test(1));

  BV2.set(BV1);
  EXPECT_TRUE(BV2.test(0));
}

TEST(CoalescingBitVectorTest, Count) {
  UBitVec::Allocator Alloc;
  UBitVec BV(Alloc);
  EXPECT_EQ(BV.count(), 0u);
  BV.set(0);
  EXPECT_EQ(BV.count(), 1u);
  BV.set({11, 12, 13, 14, 15});
  EXPECT_EQ(BV.count(), 6u);
}

TEST(CoalescingBitVectorTest, ClearAndEmpty) {
  UBitVec::Allocator Alloc;
  UBitVec BV(Alloc);
  EXPECT_TRUE(BV.empty());
  BV.set(1);
  EXPECT_FALSE(BV.empty());
  BV.clear();
  EXPECT_TRUE(BV.empty());
}

TEST(CoalescingBitVector, Copy) {
  UBitVec::Allocator Alloc;
  UBitVec BV1(Alloc);
  BV1.set(0);
  UBitVec BV2 = BV1;
  EXPECT_TRUE(elementsMatch(BV1, {0}));
  EXPECT_TRUE(elementsMatch(BV2, {0}));
  BV2.set(5);
  BV2 = BV1;
  EXPECT_TRUE(elementsMatch(BV1, {0}));
  EXPECT_TRUE(elementsMatch(BV2, {0}));
}

TEST(CoalescingBitVectorTest, Iterators) {
  UBitVec::Allocator Alloc;
  UBitVec BV(Alloc);

  BV.set({0, 1, 2});

  auto It = BV.begin();
  EXPECT_TRUE(It == BV.begin());
  EXPECT_EQ(*It, 0u);
  ++It;
  EXPECT_EQ(*It, 1u);
  ++It;
  EXPECT_EQ(*It, 2u);
  ++It;
  EXPECT_TRUE(It == BV.end());
  EXPECT_TRUE(BV.end() == BV.end());

  It = BV.begin();
  EXPECT_TRUE(It == BV.begin());
  auto ItCopy = It++;
  EXPECT_TRUE(ItCopy == BV.begin());
  EXPECT_EQ(*ItCopy, 0u);
  EXPECT_EQ(*It, 1u);

  EXPECT_TRUE(elementsMatch(BV, {0, 1, 2}));

  BV.set({4, 5, 6});
  EXPECT_TRUE(elementsMatch(BV, {0, 1, 2, 4, 5, 6}));

  BV.set(3);
  EXPECT_TRUE(elementsMatch(BV, {0, 1, 2, 3, 4, 5, 6}));

  BV.set(10);
  EXPECT_TRUE(elementsMatch(BV, {0, 1, 2, 3, 4, 5, 6, 10}));

  // Should be able to reset unset bits.
  BV.reset(3);
  BV.reset(3);
  BV.reset(20000);
  BV.set({1000, 1001, 1002});
  EXPECT_TRUE(elementsMatch(BV, {0, 1, 2, 4, 5, 6, 10, 1000, 1001, 1002}));

  auto It1 = BV.begin();
  EXPECT_TRUE(It1 == BV.begin());
  EXPECT_TRUE(++It1 == ++BV.begin());
  EXPECT_TRUE(It1 != BV.begin());
  EXPECT_TRUE(It1 != BV.end());
}

TEST(CoalescingBitVectorTest, Reset) {
  UBitVec::Allocator Alloc;
  UBitVec BV(Alloc);

  BV.set(0);
  EXPECT_TRUE(BV.test(0));
  BV.reset(0);
  EXPECT_FALSE(BV.test(0));

  BV.clear();
  BV.set({1, 2, 3});
  BV.reset(1);
  EXPECT_TRUE(elementsMatch(BV, {2, 3}));

  BV.clear();
  BV.set({1, 2, 3});
  BV.reset(2);
  EXPECT_TRUE(elementsMatch(BV, {1, 3}));

  BV.clear();
  BV.set({1, 2, 3});
  BV.reset(3);
  EXPECT_TRUE(elementsMatch(BV, {1, 2}));
}

TEST(CoalescingBitVectorTest, Comparison) {
  UBitVec::Allocator Alloc;
  UBitVec BV1(Alloc);
  UBitVec BV2(Alloc);

  // Single interval.
  BV1.set({1, 2, 3});
  BV2.set({1, 2, 3});
  EXPECT_EQ(BV1, BV2);
  EXPECT_FALSE(BV1 != BV2);

  // Different number of intervals.
  BV1.clear();
  BV2.clear();
  BV1.set({1, 2, 3});
  EXPECT_NE(BV1, BV2);

  // Multiple intervals.
  BV1.clear();
  BV2.clear();
  BV1.set({1, 2, 11, 12});
  BV2.set({1, 2, 11, 12});
  EXPECT_EQ(BV1, BV2);
  BV2.reset(1);
  EXPECT_NE(BV1, BV2);
  BV2.set(1);
  BV2.reset(11);
  EXPECT_NE(BV1, BV2);
}

// A simple implementation of set union, used to double-check the human
// "expected" answer.
UBitVec simpleUnion(UBitVec::Allocator &Alloc, const UBitVec &LHS,
                    const UBitVec &RHS) {
  UBitVec Union(Alloc);
  for (unsigned Bit : LHS)
    Union.test_and_set(Bit);
  for (unsigned Bit : RHS)
    Union.test_and_set(Bit);
  return Union;
}

TEST(CoalescingBitVectorTest, Union) {
  UBitVec::Allocator Alloc;

  // Check that after doing LHS |= RHS, LHS == Expected.
  auto unionIs = [&](std::initializer_list<unsigned> LHS,
                     std::initializer_list<unsigned> RHS,
                     std::initializer_list<unsigned> Expected) {
    UBitVec BV1(Alloc);
    BV1.set(LHS);
    UBitVec BV2(Alloc);
    BV2.set(RHS);
    const UBitVec &DoubleCheckedExpected = simpleUnion(Alloc, BV1, BV2);
    ASSERT_TRUE(elementsMatch(DoubleCheckedExpected, Expected));
    BV1 |= BV2;
    ASSERT_TRUE(elementsMatch(BV1, Expected));
  };

  // Check that "LHS |= RHS" and "RHS |= LHS" both produce the expected result.
  auto testUnionSymmetrically = [&](std::initializer_list<unsigned> LHS,
                     std::initializer_list<unsigned> RHS,
                     std::initializer_list<unsigned> Expected) {
    unionIs(LHS, RHS, Expected);
    unionIs(RHS, LHS, Expected);
  };

  // Empty LHS.
  testUnionSymmetrically({}, {1, 2, 3}, {1, 2, 3});

  // Empty RHS.
  testUnionSymmetrically({1, 2, 3}, {}, {1, 2, 3});

  // Full overlap.
  testUnionSymmetrically({1}, {1}, {1});
  testUnionSymmetrically({1, 2, 11, 12}, {1, 2, 11, 12}, {1, 2, 11, 12});

  // Sliding window: fix {2, 3, 4} as the LHS, and slide a window before/after
  // it. Repeat this swapping LHS and RHS.
  testUnionSymmetrically({2, 3, 4}, {1, 2, 3}, {1, 2, 3, 4});
  testUnionSymmetrically({2, 3, 4}, {2, 3, 4}, {2, 3, 4});
  testUnionSymmetrically({2, 3, 4}, {3, 4, 5}, {2, 3, 4, 5});
  testUnionSymmetrically({1, 2, 3}, {2, 3, 4}, {1, 2, 3, 4});
  testUnionSymmetrically({3, 4, 5}, {2, 3, 4}, {2, 3, 4, 5});

  // Multiple overlaps, but at least one of the overlaps forces us to split an
  // interval (and possibly both do). For ease of understanding, fix LHS to be
  // {1, 2, 11, 12}, but vary RHS.
  testUnionSymmetrically({1, 2, 11, 12}, {1}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {2}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {11}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {12}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {1, 11}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {1, 12}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {2, 11}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {2, 12}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {1, 2, 11}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {1, 2, 12}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {1, 11, 12}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {2, 11, 12}, {1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {0, 11, 12}, {0, 1, 2, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {3, 11, 12}, {1, 2, 3, 11, 12});
  testUnionSymmetrically({1, 2, 11, 12}, {1, 11, 13}, {1, 2, 11, 12, 13});
  testUnionSymmetrically({1, 2, 11, 12}, {1, 10, 11}, {1, 2, 10, 11, 12});

  // Partial overlap, but the existing interval covers future overlaps.
  testUnionSymmetrically({1, 2, 3, 4, 5, 6, 7, 8}, {2, 3, 4, 6, 7},
                         {1, 2, 3, 4, 5, 6, 7, 8});
  testUnionSymmetrically({1, 2, 3, 4, 5, 6, 7, 8}, {2, 3, 7, 8, 9},
                         {1, 2, 3, 4, 5, 6, 7, 8, 9});

  // More partial overlaps.
  testUnionSymmetrically({1, 2, 3, 4, 5}, {0, 1, 2, 4, 5, 6},
                         {0, 1, 2, 3, 4, 5, 6});
  testUnionSymmetrically({2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4});
  testUnionSymmetrically({3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4});
  testUnionSymmetrically({1, 2}, {1, 2, 3, 4}, {1, 2, 3, 4});
  testUnionSymmetrically({0, 1}, {1, 2, 3, 4}, {0, 1, 2, 3, 4});

  // Merge non-overlapping.
  testUnionSymmetrically({0, 1}, {2, 3}, {0, 1, 2, 3});
  testUnionSymmetrically({0, 3}, {1, 2}, {0, 1, 2, 3});
}

// A simple implementation of set intersection, used to double-check the
// human "expected" answer.
UBitVec simpleIntersection(UBitVec::Allocator &Alloc, const UBitVec &LHS,
                           const UBitVec &RHS) {
  UBitVec Intersection(Alloc);
  for (unsigned Bit : LHS)
    if (RHS.test(Bit))
      Intersection.set(Bit);
  return Intersection;
}

TEST(CoalescingBitVectorTest, Intersection) {
  UBitVec::Allocator Alloc;

  // Check that after doing LHS &= RHS, LHS == Expected.
  auto intersectionIs = [&](std::initializer_list<unsigned> LHS,
                            std::initializer_list<unsigned> RHS,
                            std::initializer_list<unsigned> Expected) {
    UBitVec BV1(Alloc);
    BV1.set(LHS);
    UBitVec BV2(Alloc);
    BV2.set(RHS);
    const UBitVec &DoubleCheckedExpected = simpleIntersection(Alloc, BV1, BV2);
    ASSERT_TRUE(elementsMatch(DoubleCheckedExpected, Expected));
    BV1 &= BV2;
    ASSERT_TRUE(elementsMatch(BV1, Expected));
  };

  // Check that "LHS &= RHS" and "RHS &= LHS" both produce the expected result.
  auto testIntersectionSymmetrically = [&](std::initializer_list<unsigned> LHS,
                     std::initializer_list<unsigned> RHS,
                     std::initializer_list<unsigned> Expected) {
    intersectionIs(LHS, RHS, Expected);
    intersectionIs(RHS, LHS, Expected);
  };

  // Empty case, one-element case.
  testIntersectionSymmetrically({}, {}, {});
  testIntersectionSymmetrically({1}, {1}, {1});
  testIntersectionSymmetrically({1}, {2}, {});

  // Exact overlaps cases: single overlap and multiple overlaps.
  testIntersectionSymmetrically({1, 2}, {1, 2}, {1, 2});
  testIntersectionSymmetrically({1, 2, 11, 12}, {1, 2, 11, 12}, {1, 2, 11, 12});

  // Sliding window: fix {2, 3, 4} as the LHS, and slide a window before/after
  // it.
  testIntersectionSymmetrically({2, 3, 4}, {1, 2, 3}, {2, 3});
  testIntersectionSymmetrically({2, 3, 4}, {2, 3, 4}, {2, 3, 4});
  testIntersectionSymmetrically({2, 3, 4}, {3, 4, 5}, {3, 4});

  // No overlap, but we have multiple intervals.
  testIntersectionSymmetrically({1, 2, 11, 12}, {3, 4, 13, 14}, {});

  // Multiple overlaps, but at least one of the overlaps forces us to split an
  // interval (and possibly both do). For ease of understanding, fix LHS to be
  // {1, 2, 11, 12}, but vary RHS.
  testIntersectionSymmetrically({1, 2, 11, 12}, {1}, {1});
  testIntersectionSymmetrically({1, 2, 11, 12}, {2}, {2});
  testIntersectionSymmetrically({1, 2, 11, 12}, {11}, {11});
  testIntersectionSymmetrically({1, 2, 11, 12}, {12}, {12});
  testIntersectionSymmetrically({1, 2, 11, 12}, {1, 11}, {1, 11});
  testIntersectionSymmetrically({1, 2, 11, 12}, {1, 12}, {1, 12});
  testIntersectionSymmetrically({1, 2, 11, 12}, {2, 11}, {2, 11});
  testIntersectionSymmetrically({1, 2, 11, 12}, {2, 12}, {2, 12});
  testIntersectionSymmetrically({1, 2, 11, 12}, {1, 2, 11}, {1, 2, 11});
  testIntersectionSymmetrically({1, 2, 11, 12}, {1, 2, 12}, {1, 2, 12});
  testIntersectionSymmetrically({1, 2, 11, 12}, {1, 11, 12}, {1, 11, 12});
  testIntersectionSymmetrically({1, 2, 11, 12}, {2, 11, 12}, {2, 11, 12});
  testIntersectionSymmetrically({1, 2, 11, 12}, {0, 11, 12}, {11, 12});
  testIntersectionSymmetrically({1, 2, 11, 12}, {3, 11, 12}, {11, 12});
  testIntersectionSymmetrically({1, 2, 11, 12}, {1, 11, 13}, {1, 11});
  testIntersectionSymmetrically({1, 2, 11, 12}, {1, 10, 11}, {1, 11});

  // Partial overlap, but the existing interval covers future overlaps.
  testIntersectionSymmetrically({1, 2, 3, 4, 5, 6, 7, 8}, {2, 3, 4, 6, 7},
                                {2, 3, 4, 6, 7});
}

// A simple implementation of set intersection-with-complement, used to
// double-check the human "expected" answer.
UBitVec simpleIntersectionWithComplement(UBitVec::Allocator &Alloc,
                                         const UBitVec &LHS,
                                         const UBitVec &RHS) {
  UBitVec Intersection(Alloc);
  for (unsigned Bit : LHS)
    if (!RHS.test(Bit))
      Intersection.set(Bit);
  return Intersection;
}

TEST(CoalescingBitVectorTest, IntersectWithComplement) {
  UBitVec::Allocator Alloc;

  // Check that after doing LHS.intersectWithComplement(RHS), LHS == Expected.
  auto intersectionWithComplementIs =
      [&](std::initializer_list<unsigned> LHS,
          std::initializer_list<unsigned> RHS,
          std::initializer_list<unsigned> Expected) {
        UBitVec BV1(Alloc);
        BV1.set(LHS);
        UBitVec BV2(Alloc);
        BV2.set(RHS);
        const UBitVec &DoubleCheckedExpected =
            simpleIntersectionWithComplement(Alloc, BV1, BV2);
        ASSERT_TRUE(elementsMatch(DoubleCheckedExpected, Expected));
        BV1.intersectWithComplement(BV2);
        ASSERT_TRUE(elementsMatch(BV1, Expected));
      };

  // Empty case, one-element case.
  intersectionWithComplementIs({}, {}, {});
  intersectionWithComplementIs({1}, {1}, {});
  intersectionWithComplementIs({1}, {2}, {1});

  // Exact overlaps cases: single overlap and multiple overlaps.
  intersectionWithComplementIs({1, 2}, {1, 2}, {});
  intersectionWithComplementIs({1, 2, 11, 12}, {1, 2, 11, 12}, {});

  // Sliding window: fix {2, 3, 4} as the LHS, and slide a window before/after
  // it. Repeat this swapping LHS and RHS.
  intersectionWithComplementIs({2, 3, 4}, {1, 2, 3}, {4});
  intersectionWithComplementIs({2, 3, 4}, {2, 3, 4}, {});
  intersectionWithComplementIs({2, 3, 4}, {3, 4, 5}, {2});
  intersectionWithComplementIs({1, 2, 3}, {2, 3, 4}, {1});
  intersectionWithComplementIs({3, 4, 5}, {2, 3, 4}, {5});

  // No overlap, but we have multiple intervals.
  intersectionWithComplementIs({1, 2, 11, 12}, {3, 4, 13, 14}, {1, 2, 11, 12});

  // Multiple overlaps. For ease of understanding, fix LHS to be
  // {1, 2, 11, 12}, but vary RHS.
  intersectionWithComplementIs({1, 2, 11, 12}, {1}, {2, 11, 12});
  intersectionWithComplementIs({1, 2, 11, 12}, {2}, {1, 11, 12});
  intersectionWithComplementIs({1, 2, 11, 12}, {11}, {1, 2, 12});
  intersectionWithComplementIs({1, 2, 11, 12}, {12}, {1, 2, 11});
  intersectionWithComplementIs({1, 2, 11, 12}, {1, 11}, {2, 12});
  intersectionWithComplementIs({1, 2, 11, 12}, {1, 12}, {2, 11});
  intersectionWithComplementIs({1, 2, 11, 12}, {2, 11}, {1, 12});
  intersectionWithComplementIs({1, 2, 11, 12}, {2, 12}, {1, 11});
  intersectionWithComplementIs({1, 2, 11, 12}, {1, 2, 11}, {12});
  intersectionWithComplementIs({1, 2, 11, 12}, {1, 2, 12}, {11});
  intersectionWithComplementIs({1, 2, 11, 12}, {1, 11, 12}, {2});
  intersectionWithComplementIs({1, 2, 11, 12}, {2, 11, 12}, {1});
  intersectionWithComplementIs({1, 2, 11, 12}, {0, 11, 12}, {1, 2});
  intersectionWithComplementIs({1, 2, 11, 12}, {3, 11, 12}, {1, 2});
  intersectionWithComplementIs({1, 2, 11, 12}, {1, 11, 13}, {2, 12});
  intersectionWithComplementIs({1, 2, 11, 12}, {1, 10, 11}, {2, 12});

  // Partial overlap, but the existing interval covers future overlaps.
  intersectionWithComplementIs({1, 2, 3, 4, 5, 6, 7, 8}, {2, 3, 4, 6, 7},
                               {1, 5, 8});
}

TEST(CoalescingBitVectorTest, FindLowerBound) {
  U64BitVec::Allocator Alloc;
  U64BitVec BV(Alloc);
  uint64_t BigNum1 = uint64_t(1) << 32;
  uint64_t BigNum2 = (uint64_t(1) << 33) + 1;
  EXPECT_TRUE(BV.find(BigNum1) == BV.end());
  BV.set(BigNum1);
  auto Find1 = BV.find(BigNum1);
  EXPECT_EQ(*Find1, BigNum1);
  BV.set(BigNum2);
  auto Find2 = BV.find(BigNum1);
  EXPECT_EQ(*Find2, BigNum1);
  auto Find3 = BV.find(BigNum2);
  EXPECT_EQ(*Find3, BigNum2);
  BV.reset(BigNum1);
  auto Find4 = BV.find(BigNum1);
  EXPECT_EQ(*Find4, BigNum2);

  BV.clear();
  BV.set({1, 2, 3});
  EXPECT_EQ(*BV.find(2), 2u);
  EXPECT_EQ(*BV.find(3), 3u);
}

TEST(CoalescingBitVectorTest, AdvanceToLowerBound) {
  U64BitVec::Allocator Alloc;
  U64BitVec BV(Alloc);
  uint64_t BigNum1 = uint64_t(1) << 32;
  uint64_t BigNum2 = (uint64_t(1) << 33) + 1;

  auto advFromBegin = [&](uint64_t To) -> U64BitVec::const_iterator {
    auto It = BV.begin();
    It.advanceToLowerBound(To);
    return It;
  };

  EXPECT_TRUE(advFromBegin(BigNum1) == BV.end());
  BV.set(BigNum1);
  auto Find1 = advFromBegin(BigNum1);
  EXPECT_EQ(*Find1, BigNum1);
  BV.set(BigNum2);
  auto Find2 = advFromBegin(BigNum1);
  EXPECT_EQ(*Find2, BigNum1);
  auto Find3 = advFromBegin(BigNum2);
  EXPECT_EQ(*Find3, BigNum2);
  BV.reset(BigNum1);
  auto Find4 = advFromBegin(BigNum1);
  EXPECT_EQ(*Find4, BigNum2);

  BV.clear();
  BV.set({1, 2, 3});
  EXPECT_EQ(*advFromBegin(2), 2u);
  EXPECT_EQ(*advFromBegin(3), 3u);
  auto It = BV.begin();
  It.advanceToLowerBound(0);
  EXPECT_EQ(*It, 1u);
  It.advanceToLowerBound(100);
  EXPECT_TRUE(It == BV.end());
  It.advanceToLowerBound(100);
  EXPECT_TRUE(It == BV.end());
}

TEST(CoalescingBitVectorTest, Print) {
  std::string S;
  {
    raw_string_ostream OS(S);
    UBitVec::Allocator Alloc;
    UBitVec BV(Alloc);
    BV.set({1});
    BV.print(OS);

    BV.clear();
    BV.set({1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    BV.print(OS);
  }
  EXPECT_EQ(S, "{[1]}"
               "{[1][11, 20]}");
}

} // namespace
