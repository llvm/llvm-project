//===------ ADT/SparseSetTest.cpp - SparseSet unit tests -  -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SparseSet.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

typedef SparseSet<unsigned> USet;

// Empty set tests.
TEST(SparseSetTest, EmptySet) {
  USet Set;
  EXPECT_TRUE(Set.empty());
  EXPECT_TRUE(Set.begin() == Set.end());
  EXPECT_EQ(0u, Set.size());

  Set.setUniverse(10);

  // Lookups on empty set.
  EXPECT_FALSE(Set.contains(0));
  EXPECT_FALSE(Set.contains(9));

  // Same thing on a const reference.
  const USet &CSet = Set;
  EXPECT_TRUE(CSet.empty());
  EXPECT_TRUE(CSet.begin() == CSet.end());
  EXPECT_EQ(0u, CSet.size());
  EXPECT_FALSE(CSet.contains(0));
  USet::const_iterator I = CSet.find(5);
  EXPECT_TRUE(I == CSet.end());
}

// Single entry set tests.
TEST(SparseSetTest, SingleEntrySet) {
  USet Set;
  Set.setUniverse(10);
  std::pair<USet::iterator, bool> IP = Set.insert(5);
  EXPECT_TRUE(IP.second);
  EXPECT_TRUE(IP.first == Set.begin());

  EXPECT_FALSE(Set.empty());
  EXPECT_FALSE(Set.begin() == Set.end());
  EXPECT_TRUE(Set.begin() + 1 == Set.end());
  EXPECT_EQ(1u, Set.size());

  EXPECT_FALSE(Set.contains(0));
  EXPECT_FALSE(Set.contains(9));
  EXPECT_TRUE(Set.contains(5));

  EXPECT_FALSE(Set.count(0));
  EXPECT_TRUE(Set.count(5));

  // Redundant insert.
  IP = Set.insert(5);
  EXPECT_FALSE(IP.second);
  EXPECT_TRUE(IP.first == Set.begin());

  // Erase non-existent element.
  EXPECT_FALSE(Set.erase(1));
  EXPECT_EQ(1u, Set.size());
  EXPECT_EQ(5u, *Set.begin());

  // Erase iterator.
  USet::iterator I = Set.find(5);
  EXPECT_TRUE(I == Set.begin());
  I = Set.erase(I);
  EXPECT_FALSE(Set.contains(5));
  EXPECT_TRUE(I == Set.end());
  EXPECT_TRUE(Set.empty());
}

// Multiple entry set tests.
TEST(SparseSetTest, MultipleEntrySet) {
  USet Set;
  Set.setUniverse(10);

  Set.insert(5);
  Set.insert(3);
  Set.insert(2);
  Set.insert(1);
  Set.insert(4);
  EXPECT_EQ(5u, Set.size());

  // Without deletions, iteration order == insertion order.
  USet::const_iterator I = Set.begin();
  EXPECT_EQ(5u, *I);
  ++I;
  EXPECT_EQ(3u, *I);
  ++I;
  EXPECT_EQ(2u, *I);
  ++I;
  EXPECT_EQ(1u, *I);
  ++I;
  EXPECT_EQ(4u, *I);
  ++I;
  EXPECT_TRUE(I == Set.end());

  // Redundant insert.
  std::pair<USet::iterator, bool> IP = Set.insert(3);
  EXPECT_FALSE(IP.second);
  EXPECT_TRUE(IP.first == Set.begin() + 1);

  // Erase last element by key.
  EXPECT_TRUE(Set.erase(4));
  EXPECT_EQ(4u, Set.size());
  EXPECT_FALSE(Set.count(4));
  EXPECT_FALSE(Set.erase(4));
  EXPECT_EQ(4u, Set.size());
  EXPECT_FALSE(Set.count(4));

  // Erase first element by key.
  EXPECT_TRUE(Set.count(5));
  EXPECT_TRUE(Set.find(5) == Set.begin());
  EXPECT_TRUE(Set.erase(5));
  EXPECT_EQ(3u, Set.size());
  EXPECT_FALSE(Set.count(5));
  EXPECT_FALSE(Set.erase(5));
  EXPECT_EQ(3u, Set.size());
  EXPECT_FALSE(Set.count(5));

  Set.insert(6);
  Set.insert(7);
  EXPECT_EQ(5u, Set.size());

  // Erase last element by iterator.
  I = Set.erase(Set.end() - 1);
  EXPECT_TRUE(I == Set.end());
  EXPECT_EQ(4u, Set.size());

  // Erase second element by iterator.
  I = Set.erase(Set.begin() + 1);
  EXPECT_TRUE(I == Set.begin() + 1);

  // Clear and resize the universe.
  Set.clear();
  EXPECT_FALSE(Set.count(5));
  Set.setUniverse(1000);

  // Add more than 256 elements.
  for (unsigned i = 100; i != 800; ++i)
    Set.insert(i);

  for (unsigned i = 0; i != 10; ++i)
    Set.erase(i);

  for (unsigned i = 100; i != 800; ++i)
    EXPECT_TRUE(Set.count(i));

  EXPECT_FALSE(Set.count(99));
  EXPECT_FALSE(Set.count(800));
  EXPECT_EQ(700u, Set.size());
}

struct Alt {
  unsigned Value;
  explicit Alt(unsigned x) : Value(x) {}
  unsigned getSparseSetIndex() const { return Value - 1000; }
};

TEST(SparseSetTest, AltStructSet) {
  typedef SparseSet<Alt> ASet;
  ASet Set;
  Set.setUniverse(10);
  Set.insert(Alt(1005));

  ASet::iterator I = Set.find(5);
  ASSERT_TRUE(I == Set.begin());
  EXPECT_EQ(1005u, I->Value);

  Set.insert(Alt(1006));
  Set.insert(Alt(1006));
  I = Set.erase(Set.begin());
  ASSERT_TRUE(I == Set.begin());
  EXPECT_EQ(1006u, I->Value);

  EXPECT_FALSE(Set.erase(5));
  EXPECT_TRUE(Set.erase(6));
}

TEST(SparseSetTest, PopBack) {
  USet Set;
  const unsigned UpperBound = 300;
  Set.setUniverse(UpperBound);
  for (unsigned i = 0; i < UpperBound; ++i)
    Set.insert(i);

  // Make sure pop back returns the values in the reverse order we
  // inserted them.
  unsigned Expected = UpperBound;
  while (!Set.empty())
    ASSERT_TRUE(--Expected == Set.pop_back_val());

  // Insert again the same elements in the sparse set and make sure
  // each insertion actually inserts the elements. I.e., check
  // that the underlying data structure are properly cleared.
  for (unsigned i = 0; i < UpperBound; ++i)
    ASSERT_TRUE(Set.insert(i).second);
}
} // namespace
