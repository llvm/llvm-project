//===- llvm/unittest/ADT/FoldingSetTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FoldingSet unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FoldingSet.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;
using testing::ElementsAre;
using testing::IsEmpty;
using testing::SizeIs;
using testing::UnorderedElementsAre;

namespace {

// Unaligned string test.
TEST(FoldingSetTest, UnalignedStringTest) {
  SCOPED_TRACE("UnalignedStringTest");

  FoldingSetNodeID a, b;
  // An aligned string.
  std::string str1 = "a test string";
  a.AddString(str1);

  // An unaligned string.
  std::string str2 = ">" + str1;
  b.AddString(str2.c_str() + 1);

  EXPECT_EQ(a.ComputeHash(), b.ComputeHash());
}

TEST(FoldingSetTest, LongLongComparison) {
  struct LongLongContainer : FoldingSetNode {
    unsigned long long A, B;
    LongLongContainer(unsigned long long A, unsigned long long B)
        : A(A), B(B) {}
    void Profile(FoldingSetNodeID &ID) const {
      ID.AddInteger(A);
      ID.AddInteger(B);
    }
  };

  LongLongContainer C1((1ULL << 32) + 1, 1ULL);
  LongLongContainer C2(1ULL, (1ULL << 32) + 1);

  FoldingSet<LongLongContainer> Set;

  EXPECT_EQ(&C1, Set.GetOrInsertNode(&C1));
  EXPECT_EQ(&C2, Set.GetOrInsertNode(&C2));
  EXPECT_EQ(2U, Set.size());
}

struct TrivialPair : public FoldingSetNode {
  unsigned Key = 0;
  unsigned Value = 0;
  TrivialPair(unsigned K, unsigned V) : FoldingSetNode(), Key(K), Value(V) {}

  void Profile(FoldingSetNodeID &ID) const {
    ID.AddInteger(Key);
    ID.AddInteger(Value);
  }

  bool operator==(const TrivialPair &RHS) const {
    return Key == RHS.Key && Value == RHS.Value;
  }
};

TEST(FoldingSetTest, IDComparison) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);

  void *InsertPos = nullptr;
  FoldingSetNodeID ID;
  T.Profile(ID);
  TrivialPair *N = Trivial.FindNodeOrInsertPos(ID, InsertPos);
  EXPECT_EQ(&T, N);
  EXPECT_EQ(nullptr, InsertPos);
}

TEST(FoldingSetTest, MissedIDComparison) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair S(100, 42);
  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);

  void *InsertPos = nullptr;
  FoldingSetNodeID ID;
  S.Profile(ID);
  TrivialPair *N = Trivial.FindNodeOrInsertPos(ID, InsertPos);
  EXPECT_EQ(nullptr, N);
  EXPECT_NE(nullptr, InsertPos);
}

TEST(FoldingSetTest, RemoveNodeThatIsPresent) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);
  EXPECT_EQ(Trivial.size(), 1U);

  bool WasThere = Trivial.RemoveNode(&T);
  EXPECT_TRUE(WasThere);
  EXPECT_EQ(0U, Trivial.size());
}

TEST(FoldingSetTest, RemoveNodeThatIsAbsent) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  bool WasThere = Trivial.RemoveNode(&T);
  EXPECT_FALSE(WasThere);
  EXPECT_EQ(0U, Trivial.size());
}

TEST(FoldingSetTest, GetOrInsertInserting) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  TrivialPair *N = Trivial.GetOrInsertNode(&T);
  EXPECT_EQ(&T, N);
}

TEST(FoldingSetTest, GetOrInsertGetting) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  TrivialPair T2(99, 42);
  Trivial.InsertNode(&T);
  TrivialPair *N = Trivial.GetOrInsertNode(&T2);
  EXPECT_EQ(&T, N);
}

TEST(FoldingSetTest, InsertAtPos) {
  FoldingSet<TrivialPair> Trivial;

  void *InsertPos = nullptr;
  TrivialPair Finder(99, 42);
  FoldingSetNodeID ID;
  Finder.Profile(ID);
  Trivial.FindNodeOrInsertPos(ID, InsertPos);

  TrivialPair T(99, 42);
  Trivial.InsertNode(&T, InsertPos);
  EXPECT_EQ(1U, Trivial.size());
}

TEST(FoldingSetTest, EmptyIsTrue) {
  FoldingSet<TrivialPair> Trivial;
  EXPECT_TRUE(Trivial.empty());
}

TEST(FoldingSetTest, EmptyIsFalse) {
  FoldingSet<TrivialPair> Trivial;
  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);
  EXPECT_FALSE(Trivial.empty());
}

TEST(FoldingSetTest, ClearOnEmpty) {
  FoldingSet<TrivialPair> Trivial;
  Trivial.clear();
  EXPECT_TRUE(Trivial.empty());
}

TEST(FoldingSetTest, ClearOnNonEmpty) {
  FoldingSet<TrivialPair> Trivial;
  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);
  Trivial.clear();
  EXPECT_TRUE(Trivial.empty());
}

TEST(FoldingSetTest, CapacityLargerThanReserve) {
  FoldingSet<TrivialPair> Trivial;
  unsigned OldCapacity = Trivial.capacity();
  Trivial.reserve(OldCapacity + 1);
  EXPECT_GE(Trivial.capacity(), OldCapacity + 1);
}

TEST(FoldingSetTest, SmallReserveChangesNothing) {
  FoldingSet<TrivialPair> Trivial;
  unsigned OldCapacity = Trivial.capacity();
  Trivial.reserve(OldCapacity - 1);
  EXPECT_EQ(Trivial.capacity(), OldCapacity);
}

TEST(FoldingSetTest, ReserveExactCapacity) {
  FoldingSet<TrivialPair> Trivial;
  unsigned OldCapacity = Trivial.capacity();
  Trivial.reserve(OldCapacity);
  EXPECT_EQ(Trivial.capacity(), OldCapacity);
}

TEST(FoldingSetTest, MoveConstructor) {
  FoldingSet<TrivialPair> A;
  TrivialPair T1(10, 100);
  TrivialPair T2(20, 200);
  A.InsertNode(&T1);
  A.InsertNode(&T2);
  EXPECT_THAT(A, SizeIs(2));

  FoldingSet<TrivialPair> B(std::move(A));
  EXPECT_THAT(B, SizeIs(2));
  EXPECT_THAT(B, testing::Not(IsEmpty()));

  void *InsertPos = nullptr;
  FoldingSetNodeID ID1, ID2;
  T1.Profile(ID1);
  T2.Profile(ID2);
  EXPECT_EQ(&T1, B.FindNodeOrInsertPos(ID1, InsertPos));
  EXPECT_EQ(&T2, B.FindNodeOrInsertPos(ID2, InsertPos));
}

TEST(FoldingSetTest, MoveAssignment) {
  FoldingSet<TrivialPair> A;
  FoldingSet<TrivialPair> B;
  TrivialPair T1(10, 100);
  TrivialPair T2(20, 200);
  TrivialPair T3(30, 300);
  B.InsertNode(&T1);
  A.InsertNode(&T2);
  A.InsertNode(&T3);

  B = std::move(A);
  EXPECT_THAT(B, SizeIs(2));
  EXPECT_THAT(B, testing::Not(IsEmpty()));

  void *InsertPos = nullptr;
  FoldingSetNodeID ID2, ID3;
  T2.Profile(ID2);
  T3.Profile(ID3);
  EXPECT_EQ(&T2, B.FindNodeOrInsertPos(ID2, InsertPos));
  EXPECT_EQ(&T3, B.FindNodeOrInsertPos(ID3, InsertPos));
}

TEST(FoldingSetTest, Iterator) {
  FoldingSet<TrivialPair> Set;
  EXPECT_EQ(Set.begin(), Set.end());

  TrivialPair T1(1, 10);
  TrivialPair T2(2, 20);
  TrivialPair T3(3, 30);
  Set.InsertNode(&T1);
  Set.InsertNode(&T2);
  Set.InsertNode(&T3);

  EXPECT_THAT(Set, UnorderedElementsAre(T1, T2, T3));

  ASSERT_NE(Set.begin(), Set.end());
  auto It = Set.begin();
  auto ItCopy = It++;
  EXPECT_NE(It, ItCopy);
}

TEST(FoldingSetTest, FoldingSetVectorBasic) {
  FoldingSetVector<TrivialPair> Vec;
  EXPECT_THAT(Vec, IsEmpty());
  EXPECT_THAT(Vec, SizeIs(0));

  TrivialPair T1(10, 100);
  TrivialPair T1Copy(10, 100);
  TrivialPair T2(20, 200);
  TrivialPair T3(30, 300);

  EXPECT_EQ(&T1, Vec.GetOrInsertNode(&T1));
  EXPECT_EQ(&T1, Vec.GetOrInsertNode(&T1Copy));
  EXPECT_THAT(Vec, SizeIs(1));

  // Insert a new node using an insertion token.
  FoldingSetNodeID ID2;
  T2.Profile(ID2);
  void *InsertPos = nullptr;
  EXPECT_EQ(nullptr, Vec.FindNodeOrInsertPos(ID2, InsertPos));
  ASSERT_NE(nullptr, InsertPos);
  Vec.InsertNode(&T2, InsertPos);
  EXPECT_THAT(Vec, SizeIs(2));

  Vec.InsertNode(&T3);
  EXPECT_THAT(Vec, SizeIs(3));
  EXPECT_THAT(Vec, testing::Not(IsEmpty()));

  // Verify deterministic iteration order matching insertion order.
  EXPECT_THAT(Vec, ElementsAre(T1, T2, T3));

  Vec.clear();
  EXPECT_THAT(Vec, IsEmpty());
  EXPECT_THAT(Vec, SizeIs(0));
}

struct TestContext {
  unsigned Value = 0;
};

struct ContextualPair : public FoldingSetNode {
  unsigned Key = 0;
  unsigned Value = 0;
  ContextualPair(unsigned K, unsigned V) : FoldingSetNode(), Key(K), Value(V) {}

  void Profile(FoldingSetNodeID &ID, TestContext Context) const {
    ID.AddInteger(Key ^ Context.Value);
    ID.AddInteger(Value ^ Context.Value);
  }

  bool operator==(const ContextualPair &RHS) const {
    return Key == RHS.Key && Value == RHS.Value;
  }
};

TEST(FoldingSetTest, ContextualFoldingSetBasic) {
  TestContext ContextVal{0xABCD};
  ContextualFoldingSet<ContextualPair, TestContext> Set(ContextVal);
  EXPECT_EQ(ContextVal.Value, Set.getContext().Value);
  EXPECT_THAT(Set, IsEmpty());
  EXPECT_THAT(Set, SizeIs(0));

  ContextualPair T1(10, 100);
  ContextualPair T1Copy(10, 100);
  ContextualPair T2(20, 200);

  EXPECT_EQ(&T1, Set.GetOrInsertNode(&T1));
  EXPECT_EQ(&T1, Set.GetOrInsertNode(&T1Copy));
  EXPECT_THAT(Set, SizeIs(1));

  // Insert a new node using an insertion token.
  void *InsertPos = nullptr;
  FoldingSetNodeID ID2;
  T2.Profile(ID2, ContextVal);
  EXPECT_EQ(nullptr, Set.FindNodeOrInsertPos(ID2, InsertPos));
  ASSERT_NE(nullptr, InsertPos);
  Set.InsertNode(&T2, InsertPos);
  EXPECT_THAT(Set, SizeIs(2));

  EXPECT_EQ(&T2, Set.FindNodeOrInsertPos(ID2, InsertPos));

  EXPECT_THAT(Set, UnorderedElementsAre(T1, T2));

  EXPECT_TRUE(Set.RemoveNode(&T1));
  EXPECT_THAT(Set, SizeIs(1));
  EXPECT_FALSE(Set.RemoveNode(&T1));

  EXPECT_THAT(Set, UnorderedElementsAre(T2));

  Set.clear();
  EXPECT_THAT(Set, IsEmpty());
  EXPECT_THAT(Set, SizeIs(0));
}

TEST(FoldingSetTest, SelfMoveAssignment) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(10, 100);
  Set.InsertNode(&T1);

  // Route through a helper lambda to test self-move aliasing without triggering
  // -Wself-move.
  auto MoveAssign = [](FoldingSet<TrivialPair> &Dest,
                       FoldingSet<TrivialPair> &&Src) {
    Dest = std::move(Src);
  };
  MoveAssign(Set, std::move(Set));

  EXPECT_EQ(1u, Set.size());
  EXPECT_FALSE(Set.empty());
}

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
TEST(FoldingSetTest, InsertInvalidatesIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1), T2(2, 2);
  Set.InsertNode(&T1);
  auto It = Set.begin();
  Set.InsertNode(&T2);
  EXPECT_DEATH((void)It->Value, "invalid iterator access");
}

TEST(FoldingSetTest, RemoveInvalidatesIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1), T2(2, 2);
  Set.InsertNode(&T1);
  Set.InsertNode(&T2);
  auto It = Set.begin();
  Set.RemoveNode(&T2);
  EXPECT_DEATH((void)It->Value, "invalid iterator access");
}

TEST(FoldingSetTest, RemoveOfAbsentNodeKeepsIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1), Absent(2, 2);
  Set.InsertNode(&T1);
  auto It = Set.begin();
  EXPECT_FALSE(Set.RemoveNode(&Absent));
  EXPECT_EQ(&T1, &*It);
}

TEST(FoldingSetTest, ClearInvalidatesIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1);
  Set.InsertNode(&T1);
  auto It = Set.begin();
  Set.clear();
  EXPECT_DEATH((void)It->Value, "invalid iterator access");
}

TEST(FoldingSetTest, MoveInvalidatesIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1);
  Set.InsertNode(&T1);
  auto It = Set.begin();
  FoldingSet<TrivialPair> Other(std::move(Set));
  EXPECT_DEATH((void)It->Value, "invalid iterator access");
}
#endif

} // namespace
