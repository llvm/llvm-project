//===- unittests/ADT/BumpPtrListTest.cpp - BumpPtrList unit tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/AllocatorList.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct CountsDestructors {
  static unsigned NumCalls;
  ~CountsDestructors() { ++NumCalls; }
};
unsigned CountsDestructors::NumCalls = 0;

struct MoveOnly {
  int V;
  explicit MoveOnly(int V) : V(V) {}
  MoveOnly() = delete;
  MoveOnly(MoveOnly &&X) { V = X.V; }
  MoveOnly(const MoveOnly &X) = delete;
  MoveOnly &operator=(MoveOnly &&X) = delete;
  MoveOnly &operator=(const MoveOnly &X) = delete;
};

struct EmplaceOnly {
  int V1, V2;
  explicit EmplaceOnly(int V1, int V2) : V1(V1), V2(V2) {}
  EmplaceOnly() = delete;
  EmplaceOnly(EmplaceOnly &&X) = delete;
  EmplaceOnly(const EmplaceOnly &X) = delete;
  EmplaceOnly &operator=(EmplaceOnly &&X) = delete;
  EmplaceOnly &operator=(const EmplaceOnly &X) = delete;
};

TEST(BumpPtrListTest, DefaultConstructor) {
  BumpPtrList<int> L;
  EXPECT_TRUE(L.empty());
}

TEST(BumpPtrListTest, pushPopBack) {
  // Build a list with push_back.
  BumpPtrList<int> L;
  int Ns[] = {1, 3, 9, 5, 7};
  for (const int N : Ns)
    L.push_back(N);

  // Use iterators to check contents.
  auto I = L.begin();
  for (int N : Ns)
    EXPECT_EQ(N, *I++);
  EXPECT_EQ(I, L.end());

  // Unbuild the list with pop_back.
  for (int N : llvm::reverse(Ns)) {
    EXPECT_EQ(N, L.back());
    L.pop_back();
  }
  EXPECT_TRUE(L.empty());
}

TEST(BumpPtrListTest, pushPopFront) {
  // Build a list with push_front.
  BumpPtrList<int> L;
  int Ns[] = {1, 3, 9, 5, 7};
  for (const int N : Ns)
    L.push_front(N);

  // Use reverse iterators to check contents.
  auto I = L.rbegin();
  for (int N : Ns)
    EXPECT_EQ(N, *I++);
  EXPECT_EQ(I, L.rend());

  // Unbuild the list with pop_front.
  for (int N : llvm::reverse(Ns)) {
    EXPECT_EQ(N, L.front());
    L.pop_front();
  }
  EXPECT_TRUE(L.empty());
}

TEST(BumpPtrListTest, pushBackMoveOnly) {
  BumpPtrList<MoveOnly> L;
  int Ns[] = {1, 3, 9, 5, 7};
  for (const int N : Ns) {
    L.push_back(MoveOnly(N));
    EXPECT_EQ(N, L.back().V);
  }
  // Instantiate with MoveOnly.
  while (!L.empty())
    L.pop_back();
}

TEST(BumpPtrListTest, pushFrontMoveOnly) {
  BumpPtrList<MoveOnly> L;
  int Ns[] = {1, 3, 9, 5, 7};
  for (const int N : Ns) {
    L.push_front(MoveOnly(N));
    EXPECT_EQ(N, L.front().V);
  }
  // Instantiate with MoveOnly.
  while (!L.empty())
    L.pop_front();
}

TEST(BumpPtrListTest, emplaceBack) {
  BumpPtrList<EmplaceOnly> L;
  int N1s[] = {1, 3, 9, 5, 7};
  int N2s[] = {7, 3, 1, 8, 2};
  for (int I = 0; I != 5; ++I) {
    L.emplace_back(N1s[I], N2s[I]);
    EXPECT_EQ(N1s[I], L.back().V1);
    EXPECT_EQ(N2s[I], L.back().V2);
  }
  // Instantiate with EmplaceOnly.
  while (!L.empty())
    L.pop_back();
}

TEST(BumpPtrListTest, emplaceFront) {
  BumpPtrList<EmplaceOnly> L;
  int N1s[] = {1, 3, 9, 5, 7};
  int N2s[] = {7, 3, 1, 8, 2};
  for (int I = 0; I != 5; ++I) {
    L.emplace_front(N1s[I], N2s[I]);
    EXPECT_EQ(N1s[I], L.front().V1);
    EXPECT_EQ(N2s[I], L.front().V2);
  }
  // Instantiate with EmplaceOnly.
  while (!L.empty())
    L.pop_front();
}

TEST(BumpPtrListTest, swap) {
  // Build two lists with different lifetimes and swap them.
  int N1s[] = {1, 3, 5, 7, 9};
  int N2s[] = {2, 4, 6, 8, 10};

  BumpPtrList<int> L1;
  L1.insert(L1.end(), std::begin(N1s), std::end(N1s));
  {
    BumpPtrList<int> L2;
    L2.insert(L2.end(), std::begin(N2s), std::end(N2s));

    // Swap the lists.
    L1.swap(L2);

    // Check L2's contents before it goes out of scope.
    auto I = L2.begin();
    for (int N : N1s)
      EXPECT_EQ(N, *I++);
    EXPECT_EQ(I, L2.end());
  }

  // Check L1's contents now that L2 is out of scope (with its allocation
  // blocks).
  auto I = L1.begin();
  for (int N : N2s)
    EXPECT_EQ(N, *I++);
  EXPECT_EQ(I, L1.end());
}

TEST(BumpPtrListTest, clear) {
  CountsDestructors::NumCalls = 0;
  CountsDestructors N;
  BumpPtrList<CountsDestructors> L;
  L.push_back(N);
  L.push_back(N);
  L.push_back(N);
  EXPECT_EQ(3u, L.size());
  EXPECT_EQ(0u, CountsDestructors::NumCalls);
  L.pop_back();
  EXPECT_EQ(1u, CountsDestructors::NumCalls);
  L.clear();
  EXPECT_EQ(3u, CountsDestructors::NumCalls);
}

TEST(BumpPtrListTest, move) {
  BumpPtrList<int> L1, L2;
  L1.push_back(1);
  L2.push_back(2);
  L1 = std::move(L2);
  EXPECT_EQ(1u, L1.size());
  EXPECT_EQ(2, L1.front());
  EXPECT_EQ(0u, L2.size());
}

TEST(BumpPtrListTest, moveCallsDestructors) {
  CountsDestructors::NumCalls = 0;
  BumpPtrList<CountsDestructors> L1, L2;
  L1.emplace_back();
  EXPECT_EQ(0u, CountsDestructors::NumCalls);
  L1 = std::move(L2);
  EXPECT_EQ(1u, CountsDestructors::NumCalls);
}

TEST(BumpPtrListTest, copy) {
  BumpPtrList<int> L1, L2;
  L1.push_back(1);
  L2.push_back(2);
  L1 = L2;
  EXPECT_EQ(1u, L1.size());
  EXPECT_EQ(2, L1.front());
  EXPECT_EQ(1u, L2.size());
  EXPECT_EQ(2, L2.front());
}

TEST(BumpPtrListTest, copyCallsDestructors) {
  CountsDestructors::NumCalls = 0;
  BumpPtrList<CountsDestructors> L1, L2;
  L1.emplace_back();
  EXPECT_EQ(0u, CountsDestructors::NumCalls);
  L1 = L2;
  EXPECT_EQ(1u, CountsDestructors::NumCalls);
}

TEST(BumpPtrListTest, resetAlloc) {
  // Resetting an empty list should work.
  BumpPtrList<int> L;

  // Resetting an empty list that has allocated should also work.
  L.resetAlloc();
  L.push_back(5);
  L.erase(L.begin());
  L.resetAlloc();

  // Resetting a non-empty list should crash.
  L.push_back(5);
#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)
  EXPECT_DEATH(L.resetAlloc(), "Cannot reset allocator if not empty");
#endif
}

} // end namespace
