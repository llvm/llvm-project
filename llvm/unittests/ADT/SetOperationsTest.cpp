//===- SetOperations.cpp - Unit tests for set operations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetOperations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <set>

using namespace llvm;

using testing::IsEmpty;

namespace {

TEST(SetOperationsTest, SetUnion) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {5, 6, 7, 8};
  // Set1 should be the union of input sets Set1 and Set2.
  std::set<int> ExpectedSet1 = {1, 2, 3, 4, 5, 6, 7, 8};
  // Set2 should not be touched.
  std::set<int> ExpectedSet2 = Set2;

  set_union(Set1, Set2);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);

  Set1.clear();
  Set2 = {1, 2};
  // Set1 should be the union of input sets Set1 and Set2, which in this case
  // will be Set2.
  ExpectedSet1 = Set2;
  // Set2 should not be touched.
  ExpectedSet2 = Set2;

  set_union(Set1, Set2);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);
}

TEST(SetOperationsTest, SetIntersect) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};
  // Set1 should be the intersection of sets Set1 and Set2.
  std::set<int> ExpectedSet1 = {3, 4};
  // Set2 should not be touched.
  std::set<int> ExpectedSet2 = Set2;

  set_intersect(Set1, Set2);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};
  // Set2 should not be touched.
  ExpectedSet2 = Set2;

  set_intersect(Set1, Set2);
  // Set1 should be the intersection of sets Set1 and Set2, which
  // is empty as they are non-overlapping.
  EXPECT_THAT(Set1, IsEmpty());
  EXPECT_EQ(ExpectedSet2, Set2);
}

TEST(SetOperationsTest, SetIntersection) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};
  std::set<int> Result;
  // Result should be the intersection of sets Set1 and Set2.
  std::set<int> ExpectedResult = {3, 4};
  // Set1 and Set2 should not be touched.
  std::set<int> ExpectedSet1 = Set1;
  std::set<int> ExpectedSet2 = Set2;

  Result = set_intersection(Set1, Set2);
  EXPECT_EQ(ExpectedResult, Result);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};
  // Set1 and Set2 should not be touched.
  ExpectedSet1 = Set1;
  ExpectedSet2 = Set2;

  Result = set_intersection(Set1, Set2);
  // Result should be the intersection of sets Set1 and Set2, which
  // is empty as they are non-overlapping.
  EXPECT_THAT(Result, IsEmpty());
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);

  Set1 = {5, 6};
  Set2 = {1, 2, 3, 4};
  // Set1 and Set2 should not be touched.
  ExpectedSet1 = Set1;
  ExpectedSet2 = Set2;

  Result = set_intersection(Set1, Set2);
  // Result should be the intersection of sets Set1 and Set2, which
  // is empty as they are non-overlapping. Test this again with the input sets
  // reversed, since the code takes a different path depending on which input
  // set is smaller.
  EXPECT_THAT(Result, IsEmpty());
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);
}

TEST(SetOperationsTest, SetDifference) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};
  std::set<int> Result;
  // Result should be Set1 - Set2, leaving only {1, 2}.
  std::set<int> ExpectedResult = {1, 2};
  // Set1 and Set2 should not be touched.
  std::set<int> ExpectedSet1 = Set1;
  std::set<int> ExpectedSet2 = Set2;

  Result = set_difference(Set1, Set2);
  EXPECT_EQ(ExpectedResult, Result);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);

  Set1 = {1, 2, 3, 4};
  Set2 = {1, 2, 3, 4};
  // Set1 and Set2 should not be touched.
  ExpectedSet1 = Set1;
  ExpectedSet2 = Set2;

  Result = set_difference(Set1, Set2);
  // Result should be Set1 - Set2, which should be empty.
  EXPECT_THAT(Result, IsEmpty());
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};
  // Result should be Set1 - Set2, which should be Set1 as they are
  // non-overlapping.
  ExpectedResult = Set1;
  // Set1 and Set2 should not be touched.
  ExpectedSet1 = Set1;
  ExpectedSet2 = Set2;

  Result = set_difference(Set1, Set2);
  EXPECT_EQ(ExpectedResult, Result);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);
}

TEST(SetOperationsTest, SetSubtract) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};
  // Set1 should get Set1 - Set2, leaving only {1, 2}.
  std::set<int> ExpectedSet1 = {1, 2};
  // Set2 should not be touched.
  std::set<int> ExpectedSet2 = Set2;

  set_subtract(Set1, Set2);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);

  Set1 = {1, 2, 3, 4};
  Set2 = {1, 2, 3, 4};
  // Set2 should not be touched.
  ExpectedSet2 = Set2;

  set_subtract(Set1, Set2);
  // Set1 should get Set1 - Set2, which should be empty.
  EXPECT_THAT(Set1, IsEmpty());
  EXPECT_EQ(ExpectedSet2, Set2);

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};
  // Set1 should get Set1 - Set2, which should be Set1 as they are
  // non-overlapping.
  ExpectedSet1 = Set1;
  // Set2 should not be touched.
  ExpectedSet2 = Set2;

  set_subtract(Set1, Set2);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);
}

TEST(SetOperationsTest, SetSubtractRemovedRemaining) {
  std::set<int> Removed, Remaining;

  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};
  // Set1 should get Set1 - Set2, leaving only {1, 2}.
  std::set<int> ExpectedSet1 = {1, 2};
  // Set2 should not be touched.
  std::set<int> ExpectedSet2 = Set2;
  // We should get back that {3, 4} from Set2 were removed from Set1, and {5, 6}
  // were not removed from Set1.
  std::set<int> ExpectedRemoved = {3, 4};
  std::set<int> ExpectedRemaining = {5, 6};

  set_subtract(Set1, Set2, Removed, Remaining);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);
  EXPECT_EQ(ExpectedRemoved, Removed);
  EXPECT_EQ(ExpectedRemaining, Remaining);

  Set1 = {1, 2, 3, 4};
  Set2 = {1, 2, 3, 4};
  Removed.clear();
  Remaining.clear();
  // Set2 should not be touched.
  ExpectedSet2 = Set2;
  // Set should get back that all of Set2 was removed from Set1, and nothing
  // left in Set2 was not removed from Set1.
  ExpectedRemoved = Set2;

  set_subtract(Set1, Set2, Removed, Remaining);
  // Set1 should get Set1 - Set2, which should be empty.
  EXPECT_THAT(Set1, IsEmpty());
  EXPECT_EQ(ExpectedSet2, Set2);
  EXPECT_EQ(ExpectedRemoved, Removed);
  EXPECT_THAT(Remaining, IsEmpty());

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};
  Removed.clear();
  Remaining.clear();
  // Set1 should get Set1 - Set2, which should be Set1 as they are
  // non-overlapping.
  ExpectedSet1 = {1, 2, 3, 4};
  // Set2 should not be touched.
  ExpectedSet2 = Set2;
  // Set should get back that none of Set2 was removed from Set1, and all
  // of Set2 was not removed from Set1.
  ExpectedRemaining = Set2;

  set_subtract(Set1, Set2, Removed, Remaining);
  EXPECT_EQ(ExpectedSet1, Set1);
  EXPECT_EQ(ExpectedSet2, Set2);
  EXPECT_THAT(Removed, IsEmpty());
  EXPECT_EQ(ExpectedRemaining, Remaining);
}

TEST(SetOperationsTest, SetIsSubset) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4};
  EXPECT_FALSE(set_is_subset(Set1, Set2));

  Set1 = {1, 2, 3, 4};
  Set2 = {1, 2, 3, 4};
  EXPECT_TRUE(set_is_subset(Set1, Set2));

  Set1 = {1, 2};
  Set2 = {1, 2, 3, 4};
  EXPECT_TRUE(set_is_subset(Set1, Set2));

  Set1 = {1, 2};
  Set2 = {3, 4};
  EXPECT_FALSE(set_is_subset(Set1, Set2));
}

} // namespace
