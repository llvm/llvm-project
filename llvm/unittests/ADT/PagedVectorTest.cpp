//===- llvm/unittest/ADT/PagedVectorTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PagedVector unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PagedVector.h"
#include "gtest/gtest.h"
#include <iterator>

namespace llvm {
TEST(PagedVectorTest, EmptyTest) {
  PagedVector<int, 10> V;
  EXPECT_EQ(V.empty(), true);
  EXPECT_EQ(V.size(), 0ULL);
  EXPECT_EQ(V.capacity(), 0ULL);
  EXPECT_EQ(V.materialized_begin().getIndex(), 0ULL);
  EXPECT_EQ(V.materialized_end().getIndex(), 0ULL);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 0LL);

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
  EXPECT_DEATH(V[0], "Index < Size");
  EXPECT_DEATH(PagedVector<int>(nullptr), "Allocator cannot be null");
#endif
}

TEST(PagedVectorTest, ExpandTest) {
  PagedVector<int, 10> V;
  V.resize(2);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 2ULL);
  EXPECT_EQ(V.capacity(), 10ULL);
  EXPECT_EQ(V.materialized_begin().getIndex(), 2ULL);
  EXPECT_EQ(V.materialized_end().getIndex(), 2ULL);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 0LL);
}

TEST(PagedVectorTest, FullPageFillingTest) {
  PagedVector<int, 10> V;
  V.resize(10);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 10ULL);
  EXPECT_EQ(V.capacity(), 10ULL);
  for (int I = 0; I < 10; ++I)
    V[I] = I;
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 10ULL);
  EXPECT_EQ(V.capacity(), 10ULL);
  EXPECT_EQ(V.materialized_begin().getIndex(), 0ULL);
  EXPECT_EQ(V.materialized_end().getIndex(), 10ULL);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 10LL);
  for (int I = 0; I < 10; ++I)
    EXPECT_EQ(V[I], I);
}

TEST(PagedVectorTest, HalfPageFillingTest) {
  PagedVector<int, 10> V;
  V.resize(5);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 5ULL);
  EXPECT_EQ(V.capacity(), 10ULL);
  for (int I = 0; I < 5; ++I)
    V[I] = I;
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 5LL);
  for (int I = 0; I < 5; ++I)
    EXPECT_EQ(V[I], I);

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
  for (int I = 5; I < 10; ++I)
    EXPECT_DEATH(V[I], "Index < Size");
#endif
}

TEST(PagedVectorTest, FillFullMultiPageTest) {
  PagedVector<int, 10> V;
  V.resize(20);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 20ULL);
  EXPECT_EQ(V.capacity(), 20ULL);
  for (int I = 0; I < 20; ++I)
    V[I] = I;
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 20LL);
  for (auto MI = V.materialized_begin(), ME = V.materialized_end(); MI != ME;
       ++MI)
    EXPECT_EQ(*MI, std::distance(V.materialized_begin(), MI));
}

TEST(PagedVectorTest, FillHalfMultiPageTest) {
  PagedVector<int, 10> V;
  V.resize(20);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 20ULL);
  EXPECT_EQ(V.capacity(), 20ULL);
  for (int I = 0; I < 5; ++I)
    V[I] = I;
  for (int I = 10; I < 15; ++I)
    V[I] = I;
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 20LL);
  for (int I = 0; I < 5; ++I)
    EXPECT_EQ(V[I], I);
  for (int I = 10; I < 15; ++I)
    EXPECT_EQ(V[I], I);
}

TEST(PagedVectorTest, FillLastMultiPageTest) {
  PagedVector<int, 10> V;
  V.resize(20);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 20ULL);
  EXPECT_EQ(V.capacity(), 20ULL);
  for (int I = 10; I < 15; ++I)
    V[I] = I;
  for (int I = 10; I < 15; ++I)
    EXPECT_EQ(V[I], I);

  // Since we fill the last page only, the materialized vector
  // should contain only the last page.
  int J = 10;
  for (auto MI = V.materialized_begin(), ME = V.materialized_end(); MI != ME;
       ++MI) {
    if (J < 15)
      EXPECT_EQ(*MI, J);
    else
      EXPECT_EQ(*MI, 0);
    ++J;
  }
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 10LL);
}

// Filling the first element of all the pages
// will allocate all of them
TEST(PagedVectorTest, FillSparseMultiPageTest) {
  PagedVector<int, 10> V;
  V.resize(100);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 100ULL);
  EXPECT_EQ(V.capacity(), 100ULL);
  for (int I = 0; I < 10; ++I)
    V[I * 10] = I;
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 100LL);
  for (int I = 0; I < 100; ++I)
    if (I % 10 == 0)
      EXPECT_EQ(V[I], I / 10);
    else
      EXPECT_EQ(V[I], 0);
}

struct TestHelper {
  int A = -1;
};

// Use this to count how many times the constructor / destructor are called
struct TestHelper2 {
  int A = -1;
  static int constructed;
  static int destroyed;

  TestHelper2() { constructed++; }
  ~TestHelper2() { destroyed++; }
};

int TestHelper2::constructed = 0;
int TestHelper2::destroyed = 0;

TEST(PagedVectorTest, FillNonTrivialConstructor) {
  PagedVector<TestHelper, 10> V;
  V.resize(10);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 10ULL);
  EXPECT_EQ(V.capacity(), 10ULL);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 0LL);
  for (int I = 0; I < 10; ++I)
    EXPECT_EQ(V[I].A, -1);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 10LL);
}

// Elements are constructed, destructed in pages, so we expect
// the number of constructed / destructed elements to be a multiple of the
// page size and the constructor is invoked when the page is actually accessed
// the first time.
TEST(PagedVectorTest, FillNonTrivialConstructorDestructor) {
  PagedVector<TestHelper2, 10> V;
  V.resize(19);
  EXPECT_EQ(TestHelper2::constructed, 0);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 19ULL);
  EXPECT_EQ(V.capacity(), 20ULL);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 0LL);
  EXPECT_EQ(V[0].A, -1);
  EXPECT_EQ(TestHelper2::constructed, 10);

  for (int I = 0; I < 10; ++I) {
    EXPECT_EQ(V[I].A, -1);
    EXPECT_EQ(TestHelper2::constructed, 10);
  }
  for (int I = 10; I < 11; ++I) {
    EXPECT_EQ(V[I].A, -1);
    EXPECT_EQ(TestHelper2::constructed, 20);
  }
  for (int I = 0; I < 19; ++I) {
    EXPECT_EQ(V[I].A, -1);
    EXPECT_EQ(TestHelper2::constructed, 20);
  }
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 19LL);
  // We initialize the whole page, not just the materialized part
  // EXPECT_EQ(TestHelper2::constructed, 20);
  V.resize(18);
  EXPECT_EQ(TestHelper2::destroyed, 0);
  V.resize(1);
  EXPECT_EQ(TestHelper2::destroyed, 10);
  V.resize(0);
  EXPECT_EQ(TestHelper2::destroyed, 20);

  // Add a few empty pages so that we can test that the destructor
  // is called only for the materialized pages
  V.resize(50);
  V[49].A = 0;
  EXPECT_EQ(TestHelper2::constructed, 30);
  EXPECT_EQ(TestHelper2::destroyed, 20);
  EXPECT_EQ(V[49].A, 0);
  V.resize(0);
  EXPECT_EQ(TestHelper2::destroyed, 30);
}

TEST(PagedVectorTest, ShrinkTest) {
  PagedVector<int, 10> V;
  V.resize(20);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 20ULL);
  EXPECT_EQ(V.capacity(), 20ULL);
  for (int I = 0; I < 20; ++I)
    V[I] = I;
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 20LL);
  V.resize(9);
  EXPECT_EQ(V.empty(), false);
  EXPECT_EQ(V.size(), 9ULL);
  EXPECT_EQ(V.capacity(), 10ULL);
  for (int I = 0; I < 9; ++I)
    EXPECT_EQ(V[I], I);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 9LL);
  V.resize(0);
  EXPECT_EQ(V.empty(), true);
  EXPECT_EQ(V.size(), 0ULL);
  EXPECT_EQ(V.capacity(), 0ULL);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 0LL);

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
  EXPECT_DEATH(V[0], "Index < Size");
#endif
}

TEST(PagedVectorTest, FunctionalityTest) {
  PagedVector<int, 10> V;
  EXPECT_EQ(V.empty(), true);

  // Next ten numbers are 10..19
  V.resize(2);
  EXPECT_EQ(V.empty(), false);
  V.resize(10);
  V.resize(20);
  V.resize(30);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 0LL);

  EXPECT_EQ(V.size(), 30ULL);
  for (int I = 0; I < 10; ++I)
    V[I] = I;
  for (int I = 0; I < 10; ++I)
    EXPECT_EQ(V[I], I);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 10LL);
  for (int I = 20; I < 30; ++I)
    V[I] = I;
  for (int I = 20; I < 30; ++I)
    EXPECT_EQ(V[I], I);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 20LL);

  for (int I = 10; I < 20; ++I)
    V[I] = I;
  for (int I = 10; I < 20; ++I)
    EXPECT_EQ(V[I], I);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 30LL);
  V.resize(35);
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 30LL);
  for (int I = 30; I < 35; ++I)
    V[I] = I;
  EXPECT_EQ(std::distance(V.materialized_begin(), V.materialized_end()), 35LL);
  EXPECT_EQ(V.size(), 35ULL);
  EXPECT_EQ(V.capacity(), 40ULL);
  V.resize(37);
  for (int I = 30; I < 37; ++I)
    V[I] = I;
  EXPECT_EQ(V.size(), 37ULL);
  EXPECT_EQ(V.capacity(), 40ULL);
  for (int I = 0; I < 37; ++I)
    EXPECT_EQ(V[I], I);

  V.resize(41);
  V[40] = 40;
  EXPECT_EQ(V.size(), 41ULL);
  EXPECT_EQ(V.capacity(), 50ULL);
  for (int I = 0; I < 36; ++I)
    EXPECT_EQ(V[I], I);

  for (int I = 37; I < 40; ++I)
    EXPECT_EQ(V[I], 0);

  V.resize(50);
  EXPECT_EQ(V.capacity(), 50ULL);
  EXPECT_EQ(V.size(), 50ULL);
  EXPECT_EQ(V[40], 40);
  V.resize(50ULL);
  V.clear();
  EXPECT_EQ(V.size(), 0ULL);
  EXPECT_EQ(V.capacity(), 0ULL);
}
} // namespace llvm
