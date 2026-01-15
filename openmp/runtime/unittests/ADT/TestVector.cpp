//===- TestVector.cpp - Tests for kmp_vector class -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp_adt.h"
#include "gtest/gtest.h"

namespace {

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, DefaultConstruction) {
  kmp_vector<int> v;
  EXPECT_EQ(v.size(), 0u);
  EXPECT_TRUE(v.empty());
}

TEST(kmp_vector_test, ConstructWithCapacity) {
  kmp_vector<int> v(10);
  EXPECT_EQ(v.size(), 0u);
  EXPECT_TRUE(v.empty());
}

TEST(kmp_vector_test, ConstructWithData) {
  int data[] = {1, 2, 3, 4, 5};
  kmp_vector<int> v(5, data, 5);

  EXPECT_EQ(v.size(), 5u);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
  EXPECT_EQ(v[3], 4);
  EXPECT_EQ(v[4], 5);
}

TEST(kmp_vector_test, ConstructWithCapacityLargerThanSize) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v(10, data, 3);

  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
}

//===----------------------------------------------------------------------===//
// Copy Semantics
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, CopyConstruction) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v1(3, data, 3);
  kmp_vector<int> v2(v1);

  EXPECT_EQ(v2.size(), 3u);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2[2], 3);

  // Modify v1, v2 should be unchanged
  v1[0] = 100;
  EXPECT_EQ(v2[0], 1);
}

TEST(kmp_vector_test, CopyAssignment) {
  int data1[] = {1, 2, 3};
  int data2[] = {4, 5};
  kmp_vector<int> v1(3, data1, 3);
  kmp_vector<int> v2(2, data2, 2);

  v2 = v1;

  EXPECT_EQ(v2.size(), 3u);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2[2], 3);
}

TEST(kmp_vector_test, SelfCopyAssignment) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v(3, data, 3);

  kmp_vector<int> &v_ref = v;
  v = v_ref; // Avoid self-assignment warning

  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], 1);
}

//===----------------------------------------------------------------------===//
// Move Semantics
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, MoveConstruction) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v1(3, data, 3);
  kmp_vector<int> v2(std::move(v1));

  EXPECT_EQ(v2.size(), 3u);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2[2], 3);

  // v1 should be empty after move
  EXPECT_EQ(v1.size(), 0u);
}

TEST(kmp_vector_test, MoveAssignment) {
  int data1[] = {1, 2, 3};
  int data2[] = {4, 5};
  kmp_vector<int> v1(3, data1, 3);
  kmp_vector<int> v2(2, data2, 2);

  v2 = std::move(v1);

  EXPECT_EQ(v2.size(), 3u);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v1.size(), 0u);
}

TEST(kmp_vector_test, SelfMoveAssignment) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v(3, data, 3);

  kmp_vector<int> &v_ref = v;
  v = std::move(v_ref); // Avoid self-move warning

  // Self-move should leave object in valid state
  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], 1);
}

//===----------------------------------------------------------------------===//
// push_back
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, PushBackToEmpty) {
  kmp_vector<int> v;

  v.push_back(42);

  EXPECT_EQ(v.size(), 1u);
  EXPECT_EQ(v[0], 42);
}

TEST(kmp_vector_test, PushBackMultiple) {
  kmp_vector<int> v;

  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  v.push_back(5);

  EXPECT_EQ(v.size(), 5u);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
  EXPECT_EQ(v[3], 4);
  EXPECT_EQ(v[4], 5);
}

TEST(kmp_vector_test, PushBackGrowth) {
  kmp_vector<int> v;

  // Push many elements to trigger multiple resizes
  for (int i = 0; i < 100; ++i) {
    v.push_back(i);
  }

  EXPECT_EQ(v.size(), 100u);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(v[i], i);
  }
}

//===----------------------------------------------------------------------===//
// clear
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, Clear) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v(3, data, 3);

  EXPECT_EQ(v.size(), 3u);

  v.clear();

  EXPECT_EQ(v.size(), 0u);
  EXPECT_TRUE(v.empty());
}

TEST(kmp_vector_test, ClearEmpty) {
  kmp_vector<int> v;

  v.clear();

  EXPECT_EQ(v.size(), 0u);
  EXPECT_TRUE(v.empty());
}

TEST(kmp_vector_test, PushBackAfterClear) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v(3, data, 3);

  v.clear();
  v.push_back(42);

  EXPECT_EQ(v.size(), 1u);
  EXPECT_EQ(v[0], 42);
}

//===----------------------------------------------------------------------===//
// empty
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, EmptyOnDefault) {
  kmp_vector<int> v;
  EXPECT_TRUE(v.empty());
}

TEST(kmp_vector_test, NotEmptyAfterPush) {
  kmp_vector<int> v;
  v.push_back(1);
  EXPECT_FALSE(v.empty());
}

TEST(kmp_vector_test, EmptyAfterClear) {
  kmp_vector<int> v;
  v.push_back(1);
  v.clear();
  EXPECT_TRUE(v.empty());
}

//===----------------------------------------------------------------------===//
// contains
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, ContainsFound) {
  int data[] = {1, 2, 3, 4, 5};
  kmp_vector<int> v(5, data, 5);

  EXPECT_TRUE(v.contains(1));
  EXPECT_TRUE(v.contains(3));
  EXPECT_TRUE(v.contains(5));
}

TEST(kmp_vector_test, ContainsNotFound) {
  int data[] = {1, 2, 3, 4, 5};
  kmp_vector<int> v(5, data, 5);

  EXPECT_FALSE(v.contains(0));
  EXPECT_FALSE(v.contains(6));
  EXPECT_FALSE(v.contains(-1));
}

TEST(kmp_vector_test, ContainsEmpty) {
  kmp_vector<int> v;

  EXPECT_FALSE(v.contains(0));
  EXPECT_FALSE(v.contains(1));
}

//===----------------------------------------------------------------------===//
// is_set_equal
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, IsSetEqualSameOrder) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v1(3, data, 3);
  kmp_vector<int> v2(3, data, 3);

  EXPECT_TRUE(v1.is_set_equal(v2));
  EXPECT_TRUE(v2.is_set_equal(v1));
}

TEST(kmp_vector_test, IsSetEqualDifferentOrder) {
  int data1[] = {1, 2, 3};
  int data2[] = {3, 1, 2};
  kmp_vector<int> v1(3, data1, 3);
  kmp_vector<int> v2(3, data2, 3);

  EXPECT_TRUE(v1.is_set_equal(v2));
  EXPECT_TRUE(v2.is_set_equal(v1));
}

TEST(kmp_vector_test, IsSetEqualDifferentSize) {
  int data1[] = {1, 2, 3};
  int data2[] = {1, 2};
  kmp_vector<int> v1(3, data1, 3);
  kmp_vector<int> v2(2, data2, 2);

  EXPECT_FALSE(v1.is_set_equal(v2));
  EXPECT_FALSE(v2.is_set_equal(v1));
}

TEST(kmp_vector_test, IsSetEqualDifferentElements) {
  int data1[] = {1, 2, 3};
  int data2[] = {1, 2, 4};
  kmp_vector<int> v1(3, data1, 3);
  kmp_vector<int> v2(3, data2, 3);

  EXPECT_FALSE(v1.is_set_equal(v2));
}

TEST(kmp_vector_test, IsSetEqualEmpty) {
  kmp_vector<int> v1;
  kmp_vector<int> v2;

  EXPECT_TRUE(v1.is_set_equal(v2));
}

TEST(kmp_vector_test, IsSetEqualWithDuplicates) {
  int data1[] = {1, 1};
  int data2[] = {1, 2};
  kmp_vector<int> v1(2, data1, 2);
  kmp_vector<int> v2(2, data2, 2);

  EXPECT_FALSE(v1.is_set_equal(v2));
  EXPECT_FALSE(v2.is_set_equal(v1));
}

TEST(kmp_vector_test, IsSetEqualWithDuplicatesSame) {
  int data1[] = {1, 1};
  int data2[] = {1};
  kmp_vector<int> v1(2, data1, 2);
  kmp_vector<int> v2(1, data2, 1);

  EXPECT_TRUE(v1.is_set_equal(v2));
  EXPECT_TRUE(v2.is_set_equal(v1));
}

//===----------------------------------------------------------------------===//
// contains with Comparator
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, ContainsWithComparator) {
  int data[] = {10, 20, 30};
  kmp_vector<int> v(3, data, 3);

  // Compare by tens digit
  auto same_tens = [](const int &a, const int &b) {
    return (a / 10) == (b / 10);
  };

  EXPECT_TRUE(v.contains(15, same_tens)); // 15/10 == 1, matches 10/10 == 1
  EXPECT_TRUE(v.contains(25, same_tens)); // 25/10 == 2, matches 20/10 == 2
  EXPECT_FALSE(v.contains(45, same_tens)); // 45/10 == 4, no match
}

TEST(kmp_vector_test, ContainsPointerWithComparator) {
  int a = 100, b = 200, c = 300;
  int *data[] = {&a, &b, &c};
  kmp_vector<int *> v(3, data, 3);

  // Comparator that compares pointed-to values
  auto deref_comp = [](int *const &pa, int *const &pb) { return *pa == *pb; };

  int x = 200;
  int *px = &x;

  // Without comparator: comparing pointers (different addresses)
  EXPECT_FALSE(v.contains(px));

  // With comparator: comparing values (200 == 200)
  EXPECT_TRUE(v.contains(px, deref_comp));
}

//===----------------------------------------------------------------------===//
// is_set_equal with Comparator
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, IsSetEqualWithComparator) {
  int data1[] = {10, 20, 30};
  int data2[] = {35, 15, 25}; // Same tens digits as data1, different order
  kmp_vector<int> v1(3, data1, 3);
  kmp_vector<int> v2(3, data2, 3);

  auto same_tens = [](const int &a, const int &b) {
    return (a / 10) == (b / 10);
  };

  // Without comparator: not equal
  EXPECT_FALSE(v1.is_set_equal(v2));

  // With comparator: equal (same tens digits)
  EXPECT_TRUE(v1.is_set_equal(v2, same_tens));
}

TEST(kmp_vector_test, IsSetEqualPointerWithComparator) {
  int a1 = 100, b1 = 200, c1 = 300;
  int a2 = 100, b2 = 200, c2 = 300;
  int *data1[] = {&a1, &b1, &c1};
  int *data2[] = {&c2, &a2, &b2}; // Same values, different order and addresses
  kmp_vector<int *> v1(3, data1, 3);
  kmp_vector<int *> v2(3, data2, 3);

  auto deref_comp = [](int *const &pa, int *const &pb) { return *pa == *pb; };

  // Without comparator: not equal (different pointers)
  EXPECT_FALSE(v1.is_set_equal(v2));

  // With comparator: equal (same pointed-to values)
  EXPECT_TRUE(v1.is_set_equal(v2, deref_comp));
}

//===----------------------------------------------------------------------===//
// Indexing
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, IndexOperator) {
  int data[] = {10, 20, 30};
  kmp_vector<int> v(3, data, 3);

  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);
}

TEST(kmp_vector_test, IndexOperatorModify) {
  int data[] = {10, 20, 30};
  kmp_vector<int> v(3, data, 3);

  v[1] = 200;

  EXPECT_EQ(v[1], 200);
}

TEST(kmp_vector_test, ConstIndexOperator) {
  int data[] = {10, 20, 30};
  const kmp_vector<int> v(3, data, 3);

  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);
}

//===----------------------------------------------------------------------===//
// Iterators
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, BeginEnd) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v(3, data, 3);

  int *begin = v.begin();
  int *end = v.end();

  EXPECT_EQ(end - begin, 3);
  EXPECT_EQ(*begin, 1);
  EXPECT_EQ(*(end - 1), 3);
}

TEST(kmp_vector_test, ConstBeginEnd) {
  int data[] = {1, 2, 3};
  const kmp_vector<int> v(3, data, 3);

  const int *begin = v.begin();
  const int *end = v.end();

  EXPECT_EQ(end - begin, 3);
  EXPECT_EQ(*begin, 1);
}

TEST(kmp_vector_test, RangeBasedFor) {
  int data[] = {1, 2, 3, 4, 5};
  kmp_vector<int> v(5, data, 5);

  int sum = 0;
  for (int x : v) {
    sum += x;
  }

  EXPECT_EQ(sum, 15);
}

TEST(kmp_vector_test, RangeBasedForModify) {
  int data[] = {1, 2, 3};
  kmp_vector<int> v(3, data, 3);

  for (int &x : v) {
    x *= 2;
  }

  EXPECT_EQ(v[0], 2);
  EXPECT_EQ(v[1], 4);
  EXPECT_EQ(v[2], 6);
}

TEST(kmp_vector_test, RangeBasedForEmpty) {
  kmp_vector<int> v;

  int count = 0;
  for (int x : v) {
    (void)x;
    count++;
  }

  EXPECT_EQ(count, 0);
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, EmptyVector) {
  kmp_vector<int> v;

  EXPECT_EQ(v.size(), 0u);
  EXPECT_EQ(v.begin(), v.end());
  EXPECT_FALSE(v.contains(0));
}

TEST(kmp_vector_test, SingleElement) {
  kmp_vector<int> v;
  v.push_back(42);

  EXPECT_EQ(v.size(), 1u);
  EXPECT_EQ(v[0], 42);
  EXPECT_TRUE(v.contains(42));
  EXPECT_EQ(v.end() - v.begin(), 1);
}

//===----------------------------------------------------------------------===//
// Different Types
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, PointerType) {
  int a = 1, b = 2, c = 3;
  int *data[] = {&a, &b, &c};
  kmp_vector<int *> v(3, data, 3);

  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(*v[0], 1);
  EXPECT_EQ(*v[1], 2);
  EXPECT_EQ(*v[2], 3);
}

TEST(kmp_vector_test, SizeTType) {
  size_t data[] = {100, 200, 300};
  kmp_vector<size_t> v(3, data, 3);

  EXPECT_EQ(v[0], 100u);
  EXPECT_EQ(v[1], 200u);
  EXPECT_EQ(v[2], 300u);
}

//===----------------------------------------------------------------------===//
// Reserve
//===----------------------------------------------------------------------===//

TEST(kmp_vector_test, Reserve) {
  kmp_vector<int> v;
  v.reserve(100);

  EXPECT_EQ(v.size(), 0u);

  // Push one element to get a valid data pointer
  v.push_back(0);
  const int *data_ptr = v.begin();

  // Should be able to push without reallocation
  for (int i = 1; i < 100; ++i) {
    v.push_back(i);
  }

  EXPECT_EQ(v.size(), 100u);
  EXPECT_EQ(v.begin(), data_ptr) << "Reallocation occurred despite reserve()";
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(v[i], i);
  }
}

} // namespace
