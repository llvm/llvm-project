//===- SetOperations.cpp - Unit tests for set operations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <set>

using namespace llvm;

using testing::IsEmpty;
using testing::UnorderedElementsAre;

namespace {

TEST(SetOperationsTest, SetUnion) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {5, 6, 7, 8};

  set_union(Set1, Set2);
  // Set1 should be the union of input sets Set1 and Set2.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2, 3, 4, 5, 6, 7, 8));
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(5, 6, 7, 8));

  Set1.clear();
  Set2 = {1, 2};

  set_union(Set1, Set2);
  // Set1 should be the union of input sets Set1 and Set2, which in this case
  // will be Set2.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2));
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(1, 2));
}

TEST(SetOperationsTest, SetIntersect) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};

  set_intersect(Set1, Set2);
  // Set1 should be the intersection of sets Set1 and Set2.
  EXPECT_THAT(Set1, UnorderedElementsAre(3, 4));
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(3, 4, 5, 6));

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};

  set_intersect(Set1, Set2);
  // Set1 should be the intersection of sets Set1 and Set2, which
  // is empty as they are non-overlapping.
  EXPECT_THAT(Set1, IsEmpty());
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(5, 6));

  // Check that set_intersect works on SetVector via remove_if.
  SmallSetVector<int, 4> SV;
  SV.insert(3);
  SV.insert(6);
  SV.insert(4);
  SV.insert(5);
  set_intersect(SV, Set2);
  // SV should contain only 6 and 5 now.
  EXPECT_THAT(SV, testing::ElementsAre(6, 5));
}

TEST(SetOperationsTest, SetIntersection) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};
  std::set<int> Result;

  Result = set_intersection(Set1, Set2);
  // Result should be the intersection of sets Set1 and Set2.
  EXPECT_THAT(Result, UnorderedElementsAre(3, 4));
  // Set1 and Set2 should not be touched.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2, 3, 4));
  EXPECT_THAT(Set2, UnorderedElementsAre(3, 4, 5, 6));

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};

  Result = set_intersection(Set1, Set2);
  // Result should be the intersection of sets Set1 and Set2, which
  // is empty as they are non-overlapping.
  EXPECT_THAT(Result, IsEmpty());
  // Set1 and Set2 should not be touched.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2, 3, 4));
  EXPECT_THAT(Set2, UnorderedElementsAre(5, 6));

  Set1 = {5, 6};
  Set2 = {1, 2, 3, 4};

  Result = set_intersection(Set1, Set2);
  // Result should be the intersection of sets Set1 and Set2, which
  // is empty as they are non-overlapping. Test this again with the input sets
  // reversed, since the code takes a different path depending on which input
  // set is smaller.
  EXPECT_THAT(Result, IsEmpty());
  // Set1 and Set2 should not be touched.
  EXPECT_THAT(Set1, UnorderedElementsAre(5, 6));
  EXPECT_THAT(Set2, UnorderedElementsAre(1, 2, 3, 4));
}

TEST(SetOperationsTest, SetDifference) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};
  std::set<int> Result;

  Result = set_difference(Set1, Set2);
  // Result should be Set1 - Set2, leaving only {1, 2}.
  EXPECT_THAT(Result, UnorderedElementsAre(1, 2));
  // Set1 and Set2 should not be touched.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2, 3, 4));
  EXPECT_THAT(Set2, UnorderedElementsAre(3, 4, 5, 6));

  Set1 = {1, 2, 3, 4};
  Set2 = {1, 2, 3, 4};

  Result = set_difference(Set1, Set2);
  // Result should be Set1 - Set2, which should be empty.
  EXPECT_THAT(Result, IsEmpty());
  // Set1 and Set2 should not be touched.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2, 3, 4));
  EXPECT_THAT(Set2, UnorderedElementsAre(1, 2, 3, 4));

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};

  Result = set_difference(Set1, Set2);
  // Result should be Set1 - Set2, which should be Set1 as they are
  // non-overlapping.
  EXPECT_THAT(Result, UnorderedElementsAre(1, 2, 3, 4));
  // Set1 and Set2 should not be touched.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2, 3, 4));
  EXPECT_THAT(Set2, UnorderedElementsAre(5, 6));
}

TEST(SetOperationsTest, SetSubtract) {
  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};

  set_subtract(Set1, Set2);
  // Set1 should get Set1 - Set2, leaving only {1, 2}.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2));
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(3, 4, 5, 6));

  Set1 = {1, 2, 3, 4};
  Set2 = {1, 2, 3, 4};

  set_subtract(Set1, Set2);
  // Set1 should get Set1 - Set2, which should be empty.
  EXPECT_THAT(Set1, IsEmpty());
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(1, 2, 3, 4));

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};

  set_subtract(Set1, Set2);
  // Set1 should get Set1 - Set2, which should be Set1 as they are
  // non-overlapping.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2, 3, 4));
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(5, 6));
}

TEST(SetOperationsTest, SetSubtractSmallPtrSet) {
  int A[4];

  // Set1.size() < Set2.size()
  SmallPtrSet<int *, 4> Set1 = {&A[0], &A[1]};
  SmallPtrSet<int *, 4> Set2 = {&A[1], &A[2], &A[3]};
  set_subtract(Set1, Set2);
  EXPECT_THAT(Set1, UnorderedElementsAre(&A[0]));

  // Set1.size() > Set2.size()
  Set1 = {&A[0], &A[1], &A[2]};
  Set2 = {&A[0], &A[2]};
  set_subtract(Set1, Set2);
  EXPECT_THAT(Set1, UnorderedElementsAre(&A[1]));
}

TEST(SetOperationsTest, SetSubtractSmallVector) {
  int A[4];

  // Set1.size() < Set2.size()
  SmallPtrSet<int *, 4> Set1 = {&A[0], &A[1]};
  SmallVector<int *> Set2 = {&A[1], &A[2], &A[3]};
  set_subtract(Set1, Set2);
  EXPECT_THAT(Set1, UnorderedElementsAre(&A[0]));

  // Set1.size() > Set2.size()
  Set1 = {&A[0], &A[1], &A[2]};
  Set2 = {&A[0], &A[2]};
  set_subtract(Set1, Set2);
  EXPECT_THAT(Set1, UnorderedElementsAre(&A[1]));
}

TEST(SetOperationsTest, SetSubtractRemovedRemaining) {
  std::set<int> Removed, Remaining;

  std::set<int> Set1 = {1, 2, 3, 4};
  std::set<int> Set2 = {3, 4, 5, 6};

  set_subtract(Set1, Set2, Removed, Remaining);
  // Set1 should get Set1 - Set2, leaving only {1, 2}.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2));
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(3, 4, 5, 6));
  // We should get back that {3, 4} from Set2 were removed from Set1, and {5, 6}
  // were not removed from Set1.
  EXPECT_THAT(Removed, UnorderedElementsAre(3, 4));
  EXPECT_THAT(Remaining, UnorderedElementsAre(5, 6));

  Set1 = {1, 2, 3, 4};
  Set2 = {1, 2, 3, 4};
  Removed.clear();
  Remaining.clear();

  set_subtract(Set1, Set2, Removed, Remaining);
  // Set1 should get Set1 - Set2, which should be empty.
  EXPECT_THAT(Set1, IsEmpty());
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(1, 2, 3, 4));
  // Set should get back that all of Set2 was removed from Set1, and nothing
  // left in Set2 was not removed from Set1.
  EXPECT_THAT(Removed, UnorderedElementsAre(1, 2, 3, 4));
  EXPECT_THAT(Remaining, IsEmpty());

  Set1 = {1, 2, 3, 4};
  Set2 = {5, 6};
  Removed.clear();
  Remaining.clear();

  set_subtract(Set1, Set2, Removed, Remaining);
  // Set1 should get Set1 - Set2, which should be Set1 as they are
  // non-overlapping.
  EXPECT_THAT(Set1, UnorderedElementsAre(1, 2, 3, 4));
  // Set2 should not be touched.
  EXPECT_THAT(Set2, UnorderedElementsAre(5, 6));
  EXPECT_THAT(Removed, IsEmpty());
  // Set should get back that none of Set2 was removed from Set1, and all
  // of Set2 was not removed from Set1.
  EXPECT_THAT(Remaining, UnorderedElementsAre(5, 6));
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
