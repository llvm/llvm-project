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
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

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

  EXPECT_EQ(a.computeHash(), b.computeHash());
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

  EXPECT_EQ(&C1, Set.getOrInsert(&C1));
  EXPECT_EQ(&C2, Set.getOrInsert(&C2));
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
  Trivial.insert(&T);

  FoldingSetInsertToken Token;
  FoldingSetNodeID ID;
  T.Profile(ID);
  TrivialPair *N = Trivial.lookup(ID, Token);
  EXPECT_EQ(&T, N);
  EXPECT_FALSE(Token);
}

TEST(FoldingSetTest, MissedIDComparison) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair S(100, 42);
  TrivialPair T(99, 42);
  Trivial.insert(&T);

  FoldingSetInsertToken Token;
  FoldingSetNodeID ID;
  S.Profile(ID);
  TrivialPair *N = Trivial.lookup(ID, Token);
  EXPECT_EQ(nullptr, N);
  EXPECT_TRUE(Token);
}

TEST(FoldingSetTest, EraseNodeThatIsPresent) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  Trivial.insert(&T);
  EXPECT_EQ(Trivial.size(), 1U);

  bool WasThere = Trivial.erase(&T);
  EXPECT_TRUE(WasThere);
  EXPECT_EQ(0U, Trivial.size());
}

TEST(FoldingSetTest, EraseNodeThatIsAbsent) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  bool WasThere = Trivial.erase(&T);
  EXPECT_FALSE(WasThere);
  EXPECT_EQ(0U, Trivial.size());
}

TEST(FoldingSetTest, TypedApi) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T(99, 42), TCopy(99, 42);
  FoldingSetNodeID ID;
  T.Profile(ID);

  FoldingSetInsertToken Token;
  EXPECT_EQ(nullptr, Set.lookup(ID, Token));
  ASSERT_TRUE(Token);
  Set.insert(&T, Token);
  EXPECT_EQ(&T, Set.lookup(ID, Token));
  EXPECT_FALSE(Token);
  EXPECT_EQ(&T, Set.getOrInsert(&TCopy));
  EXPECT_TRUE(Set.erase(&T));
  EXPECT_FALSE(Set.erase(&T));
}

TEST(FoldingSetTest, GetOrInsertInserting) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  TrivialPair *N = Trivial.getOrInsert(&T);
  EXPECT_EQ(&T, N);
}

TEST(FoldingSetTest, GetOrInsertGetting) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  TrivialPair T2(99, 42);
  Trivial.insert(&T);
  TrivialPair *N = Trivial.getOrInsert(&T2);
  EXPECT_EQ(&T, N);
}

TEST(FoldingSetTest, InsertWithToken) {
  FoldingSet<TrivialPair> Trivial;

  FoldingSetInsertToken Token;
  TrivialPair Finder(99, 42);
  FoldingSetNodeID ID;
  Finder.Profile(ID);
  Trivial.lookup(ID, Token);

  TrivialPair T(99, 42);
  Trivial.insert(&T, Token);
  EXPECT_EQ(1U, Trivial.size());
}

TEST(FoldingSetTest, EmptyIsTrue) {
  FoldingSet<TrivialPair> Trivial;
  EXPECT_TRUE(Trivial.empty());
}

TEST(FoldingSetTest, EmptyIsFalse) {
  FoldingSet<TrivialPair> Trivial;
  TrivialPair T(99, 42);
  Trivial.insert(&T);
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
  Trivial.insert(&T);
  Trivial.clear();
  EXPECT_TRUE(Trivial.empty());
}

// 48 is the most the default 64 buckets hold; 49 is one past it.
TEST(FoldingSetTest, Reserve) {
  for (unsigned Size : {0u, 1u, 2u, 48u, 49u}) {
    FoldingSet<TrivialPair> Set;
    Set.reserve(Size);

    std::vector<std::unique_ptr<TrivialPair>> Nodes;
    for (unsigned I = 0; I != Size; ++I) {
      Nodes.push_back(std::make_unique<TrivialPair>(I, I));
      Set.insert(Nodes.back().get());
    }
    ASSERT_EQ(Size, Set.size());

    for (unsigned I = 0; I != Size; ++I) {
      FoldingSetNodeID ID;
      ID.AddInteger(I);
      ID.AddInteger(I);
      FoldingSetInsertToken Token;
      EXPECT_EQ(Nodes[I].get(), Set.lookup(ID, Token));
    }
  }
}

TEST(FoldingSetTest, MoveConstructor) {
  FoldingSet<TrivialPair> A;
  TrivialPair T1(10, 100);
  TrivialPair T2(20, 200);
  A.insert(&T1);
  A.insert(&T2);
  EXPECT_THAT(A, SizeIs(2));

  FoldingSet<TrivialPair> B(std::move(A));
  EXPECT_THAT(B, SizeIs(2));
  EXPECT_THAT(B, testing::Not(IsEmpty()));

  FoldingSetInsertToken Token;
  FoldingSetNodeID ID1, ID2;
  T1.Profile(ID1);
  T2.Profile(ID2);
  EXPECT_EQ(&T1, B.lookup(ID1, Token));
  EXPECT_EQ(&T2, B.lookup(ID2, Token));
}

TEST(FoldingSetTest, MoveAssignment) {
  FoldingSet<TrivialPair> A;
  FoldingSet<TrivialPair> B;
  TrivialPair T1(10, 100);
  TrivialPair T2(20, 200);
  TrivialPair T3(30, 300);
  B.insert(&T1);
  A.insert(&T2);
  A.insert(&T3);

  B = std::move(A);
  EXPECT_THAT(B, SizeIs(2));
  EXPECT_THAT(B, testing::Not(IsEmpty()));

  FoldingSetInsertToken Token;
  FoldingSetNodeID ID2, ID3;
  T2.Profile(ID2);
  T3.Profile(ID3);
  EXPECT_EQ(&T2, B.lookup(ID2, Token));
  EXPECT_EQ(&T3, B.lookup(ID3, Token));
}

TEST(FoldingSetTest, Iterator) {
  FoldingSet<TrivialPair> Set;
  EXPECT_EQ(Set.begin(), Set.end());

  TrivialPair T1(1, 10);
  TrivialPair T2(2, 20);
  TrivialPair T3(3, 30);
  Set.insert(&T1);
  Set.insert(&T2);
  Set.insert(&T3);

  EXPECT_THAT(Set, UnorderedElementsAre(T1, T2, T3));

  ASSERT_NE(Set.begin(), Set.end());
  auto It = Set.begin();
  auto ItCopy = It++;
  EXPECT_NE(It, ItCopy);
}

// FoldingSetNode has a non-zero offset here, so operator* must adjust it.
struct NonEmptyBase {
  int Dummy = 0;
};

struct MultiplyInheritedNode : public NonEmptyBase, public FoldingSetNode {
  void Profile(FoldingSetNodeID &ID) const { ID.AddInteger(0); }
};

TEST(FoldingSetTest, IteratorMultipleInheritance) {
  FoldingSet<MultiplyInheritedNode> Set;
  MultiplyInheritedNode N;
  Set.insert(&N);

  EXPECT_EQ(&*Set.begin(), &N);
}

TEST(FoldingSetTest, FoldingSetVectorBasic) {
  FoldingSetVector<TrivialPair> Vec;
  EXPECT_THAT(Vec, IsEmpty());
  EXPECT_THAT(Vec, SizeIs(0));

  TrivialPair T1(10, 100);
  TrivialPair T1Copy(10, 100);
  TrivialPair T2(20, 200);
  TrivialPair T3(30, 300);

  EXPECT_EQ(&T1, Vec.getOrInsert(&T1));
  EXPECT_EQ(&T1, Vec.getOrInsert(&T1Copy));
  EXPECT_THAT(Vec, SizeIs(1));

  // Insert a new node using an insertion token.
  FoldingSetNodeID ID2;
  T2.Profile(ID2);
  FoldingSetInsertToken Token;
  EXPECT_EQ(nullptr, Vec.lookup(ID2, Token));
  ASSERT_TRUE(Token);
  Vec.insert(&T2, Token);
  EXPECT_THAT(Vec, SizeIs(2));

  EXPECT_EQ(&T3, Vec.getOrInsert(&T3));
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

  EXPECT_EQ(&T1, Set.getOrInsert(&T1));
  EXPECT_EQ(&T1, Set.getOrInsert(&T1Copy));
  EXPECT_THAT(Set, SizeIs(1));

  // Insert a new node using an insertion token.
  FoldingSetInsertToken Token;
  FoldingSetNodeID ID2;
  T2.Profile(ID2, ContextVal);
  EXPECT_EQ(nullptr, Set.lookup(ID2, Token));
  ASSERT_TRUE(Token);
  Set.insert(&T2, Token);
  EXPECT_THAT(Set, SizeIs(2));

  EXPECT_EQ(&T2, Set.lookup(ID2, Token));

  EXPECT_THAT(Set, UnorderedElementsAre(T1, T2));

  EXPECT_TRUE(Set.erase(&T1));
  EXPECT_THAT(Set, SizeIs(1));
  EXPECT_FALSE(Set.erase(&T1));

  EXPECT_THAT(Set, UnorderedElementsAre(T2));

  Set.clear();
  EXPECT_THAT(Set, IsEmpty());
  EXPECT_THAT(Set, SizeIs(0));
}

TEST(FoldingSetTest, SelfMoveAssignment) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(10, 100);
  Set.insert(&T1);

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

// Exercise growth and Algorithm R shifting against a reference model.
TEST(FoldingSetTest, InsertEraseStress) {
  FoldingSet<TrivialPair> Set;
  std::map<unsigned, std::unique_ptr<TrivialPair>> Model;
  std::mt19937 Rng(42);
  for (unsigned Op = 0; Op != 1000; ++Op) {
    unsigned Key = Rng() % 4096;
    FoldingSetNodeID ID;
    ID.AddInteger(Key);
    ID.AddInteger(Key);

    auto It = Model.find(Key);
    if (Rng() & 1) {
      FoldingSetInsertToken Token;
      TrivialPair *Found = Set.lookup(ID, Token);
      if (It != Model.end()) {
        ASSERT_EQ(It->second.get(), Found);
        continue;
      }
      ASSERT_EQ(nullptr, Found);
      auto N = std::make_unique<TrivialPair>(Key, Key);
      Set.insert(N.get(), Token);
      Model.emplace(Key, std::move(N));
    } else if (It != Model.end()) {
      ASSERT_TRUE(Set.erase(It->second.get()));
      ASSERT_FALSE(Set.erase(It->second.get()));
      Model.erase(It);
    }
    ASSERT_EQ(Model.size(), Set.size());
  }

  for (const auto &KV : Model) {
    FoldingSetNodeID ID;
    ID.AddInteger(KV.first);
    ID.AddInteger(KV.first);
    FoldingSetInsertToken Token;
    EXPECT_EQ(KV.second.get(), Set.lookup(ID, Token));
  }
  std::set<TrivialPair *> Visited;
  for (TrivialPair &N : Set)
    EXPECT_TRUE(Visited.insert(&N).second);
  EXPECT_EQ(Model.size(), Visited.size());
}

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
TEST(FoldingSetTest, InsertInvalidatesIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1), T2(2, 2);
  Set.insert(&T1);
  auto It = Set.begin();
  Set.insert(&T2);
  EXPECT_DEATH((void)It->Value, "invalid iterator access");
}

TEST(FoldingSetTest, RemoveInvalidatesIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1), T2(2, 2);
  Set.insert(&T1);
  Set.insert(&T2);
  auto It = Set.begin();
  Set.erase(&T2);
  EXPECT_DEATH((void)It->Value, "invalid iterator access");
}

TEST(FoldingSetTest, RemoveOfAbsentNodeKeepsIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1), Absent(2, 2);
  Set.insert(&T1);
  auto It = Set.begin();
  EXPECT_FALSE(Set.erase(&Absent));
  EXPECT_EQ(&T1, &*It);
}

TEST(FoldingSetTest, ClearInvalidatesIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1);
  Set.insert(&T1);
  auto It = Set.begin();
  Set.clear();
  EXPECT_DEATH((void)It->Value, "invalid iterator access");
}

TEST(FoldingSetTest, MoveInvalidatesIterators) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1);
  Set.insert(&T1);
  auto It = Set.begin();
  FoldingSet<TrivialPair> Other(std::move(Set));
  EXPECT_DEATH((void)It->Value, "invalid iterator access");
}

TEST(FoldingSetTest, IteratorComparability) {
  FoldingSet<TrivialPair> Set1, Set2;
  TrivialPair T1(1, 1), T2(2, 2);
  Set1.insert(&T1);
  Set2.insert(&T2);
  EXPECT_TRUE(Set1.begin() == Set1.begin());
  EXPECT_FALSE(Set1.begin() == Set1.end());
  EXPECT_DEATH((void)(Set1.begin() == Set2.begin()), "incomparable iterators");
}

TEST(FoldingSetTest, InsertInvalidatesIteratorComparison) {
  FoldingSet<TrivialPair> Set;
  TrivialPair T1(1, 1), T2(2, 2);
  Set.insert(&T1);
  auto It = Set.begin();
  Set.insert(&T2);
  EXPECT_DEATH((void)(It == Set.end()), "incomparable iterators");
}
#endif

// The insert token is a hash, not a position, so a rehash cannot stale it.
TEST(FoldingSetTest, TokenSurvivesGrowth) {
  FoldingSet<TrivialPair> Set;
  TrivialPair Late(9999, 9999);

  FoldingSetNodeID ID;
  Late.Profile(ID);
  FoldingSetInsertToken Token;
  ASSERT_EQ(nullptr, Set.lookup(ID, Token));
  ASSERT_TRUE(Token);

  // Force several rehashes while the token is held.
  std::vector<std::unique_ptr<TrivialPair>> Nodes;
  for (unsigned I = 0; I != 200; ++I) {
    Nodes.push_back(std::make_unique<TrivialPair>(I, I));
    Set.insert(Nodes.back().get());
  }

  Set.insert(&Late, Token);
  EXPECT_EQ(&Late, Set.lookup(ID, Token));
  EXPECT_FALSE(Token);
  EXPECT_EQ(201u, Set.size());
}

} // namespace
