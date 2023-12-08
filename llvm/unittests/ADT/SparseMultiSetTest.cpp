//===------ ADT/SparseSetTest.cpp - SparseSet unit tests -  -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SparseMultiSet.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

typedef SparseMultiSet<unsigned> USet;

// Empty set tests.
TEST(SparseMultiSetTest, EmptySet) {
  USet Set;
  EXPECT_TRUE(Set.empty());
  EXPECT_EQ(0u, Set.size());

  Set.setUniverse(10);

  // Lookups on empty set.
  EXPECT_TRUE(Set.find(0) == Set.end());
  EXPECT_TRUE(Set.find(9) == Set.end());

  // Same thing on a const reference.
  const USet &CSet = Set;
  EXPECT_TRUE(CSet.empty());
  EXPECT_EQ(0u, CSet.size());
  EXPECT_TRUE(CSet.find(0) == CSet.end());
  USet::const_iterator I = CSet.find(5);
  EXPECT_TRUE(I == CSet.end());
}

// Single entry set tests.
TEST(SparseMultiSetTest, SingleEntrySet) {
  USet Set;
  Set.setUniverse(10);
  USet::iterator I = Set.insert(5);
  EXPECT_TRUE(I != Set.end());
  EXPECT_TRUE(*I == 5);

  EXPECT_FALSE(Set.empty());
  EXPECT_EQ(1u, Set.size());

  EXPECT_TRUE(Set.find(0) == Set.end());
  EXPECT_TRUE(Set.find(9) == Set.end());

  EXPECT_FALSE(Set.contains(0));
  EXPECT_TRUE(Set.contains(5));

  // Extra insert.
  I = Set.insert(5);
  EXPECT_TRUE(I != Set.end());
  EXPECT_TRUE(I == ++Set.find(5));
  I--;
  EXPECT_TRUE(I == Set.find(5));

  // Erase non-existent element.
  I = Set.find(1);
  EXPECT_TRUE(I == Set.end());
  EXPECT_EQ(2u, Set.size());
  EXPECT_EQ(5u, *Set.find(5));

  // Erase iterator.
  I = Set.find(5);
  EXPECT_TRUE(I != Set.end());
  I = Set.erase(I);
  EXPECT_TRUE(I != Set.end());
  I = Set.erase(I);
  EXPECT_TRUE(I == Set.end());
  EXPECT_TRUE(Set.empty());
}

// Multiple entry set tests.
TEST(SparseMultiSetTest, MultipleEntrySet) {
  USet Set;
  Set.setUniverse(10);

  Set.insert(5);
  Set.insert(5);
  Set.insert(5);
  Set.insert(3);
  Set.insert(2);
  Set.insert(1);
  Set.insert(4);
  EXPECT_EQ(7u, Set.size());

  // Erase last element by key.
  EXPECT_TRUE(Set.erase(Set.find(4)) == Set.end());
  EXPECT_EQ(6u, Set.size());
  EXPECT_FALSE(Set.contains(4));
  EXPECT_TRUE(Set.find(4) == Set.end());

  // Erase first element by key.
  EXPECT_EQ(3u, Set.count(5));
  EXPECT_TRUE(Set.find(5) != Set.end());
  EXPECT_TRUE(Set.erase(Set.find(5)) != Set.end());
  EXPECT_EQ(5u, Set.size());
  EXPECT_EQ(2u, Set.count(5));

  Set.insert(6);
  Set.insert(7);
  EXPECT_EQ(7u, Set.size());

  // Erase tail by iterator.
  EXPECT_TRUE(Set.getTail(6) == Set.getHead(6));
  USet::iterator I = Set.erase(Set.find(6));
  EXPECT_TRUE(I == Set.end());
  EXPECT_EQ(6u, Set.size());

  // Erase tails by iterator.
  EXPECT_EQ(2u, Set.count(5));
  I = Set.getTail(5);
  I = Set.erase(I);
  EXPECT_TRUE(I == Set.end());
  --I;
  EXPECT_EQ(1u, Set.count(5));
  EXPECT_EQ(5u, *I);
  I = Set.erase(I);
  EXPECT_TRUE(I == Set.end());
  EXPECT_EQ(0u, Set.count(5));

  Set.insert(8);
  Set.insert(8);
  Set.insert(8);
  Set.insert(8);
  Set.insert(8);

  // Erase all the 8s
  EXPECT_EQ(5, std::distance(Set.getHead(8), Set.end()));
  Set.eraseAll(8);
  EXPECT_EQ(0, std::distance(Set.getHead(8), Set.end()));

  // Clear and resize the universe.
  Set.clear();
  EXPECT_EQ(0u, Set.size());
  EXPECT_FALSE(Set.contains(3));
  Set.setUniverse(1000);

  // Add more than 256 elements.
  for (unsigned i = 100; i != 800; ++i)
    Set.insert(i);

  for (unsigned i = 0; i != 10; ++i)
    Set.eraseAll(i);

  for (unsigned i = 100; i != 800; ++i)
    EXPECT_EQ(1u, Set.count(i));

  EXPECT_FALSE(Set.contains(99));
  EXPECT_FALSE(Set.contains(800));
  EXPECT_EQ(700u, Set.size());
}

// Test out iterators
TEST(SparseMultiSetTest, Iterators) {
  USet Set;
  Set.setUniverse(100);

  Set.insert(0);
  Set.insert(1);
  Set.insert(2);
  Set.insert(0);
  Set.insert(1);
  Set.insert(0);

  USet::RangePair RangePair = Set.equal_range(0);
  USet::iterator B = RangePair.first;
  USet::iterator E = RangePair.second;

  // Move the iterators around, going to end and coming back.
  EXPECT_EQ(3, std::distance(B, E));
  EXPECT_EQ(B, --(--(--E)));
  EXPECT_EQ(++(++(++E)), Set.end());
  EXPECT_EQ(B, --(--(--E)));
  EXPECT_EQ(++(++(++E)), Set.end());

  // Insert into the tail, and move around again
  Set.insert(0);
  EXPECT_EQ(B, --(--(--(--E))));
  EXPECT_EQ(++(++(++(++E))), Set.end());
  EXPECT_EQ(B, --(--(--(--E))));
  EXPECT_EQ(++(++(++(++E))), Set.end());

  // Erase a tail, and move around again
  USet::iterator Erased = Set.erase(Set.getTail(0));
  EXPECT_EQ(Erased, E);
  EXPECT_EQ(B, --(--(--E)));

  USet Set2;
  Set2.setUniverse(11);
  Set2.insert(3);
  EXPECT_TRUE(!Set2.contains(0));
  EXPECT_TRUE(!Set.contains(3));

  EXPECT_EQ(Set2.getHead(3), Set2.getTail(3));
  EXPECT_EQ(Set2.getHead(0), Set2.getTail(0));
  B = Set2.find(3);
  EXPECT_EQ(Set2.find(3), --(++B));
}

struct Alt {
  unsigned Value;
  explicit Alt(unsigned x) : Value(x) {}
  unsigned getSparseSetIndex() const { return Value - 1000; }
};

TEST(SparseMultiSetTest, AltStructSet) {
  typedef SparseMultiSet<Alt> ASet;
  ASet Set;
  Set.setUniverse(10);
  Set.insert(Alt(1005));

  ASet::iterator I = Set.find(5);
  ASSERT_TRUE(I != Set.end());
  EXPECT_EQ(1005u, I->Value);

  Set.insert(Alt(1006));
  Set.insert(Alt(1006));
  I = Set.erase(Set.find(6));
  ASSERT_TRUE(I != Set.end());
  EXPECT_EQ(1006u, I->Value);
  I = Set.erase(Set.find(6));
  ASSERT_TRUE(I == Set.end());

  EXPECT_TRUE(Set.contains(5));
  EXPECT_FALSE(Set.contains(6));
}
} // namespace
