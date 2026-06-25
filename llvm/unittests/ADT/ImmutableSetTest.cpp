//===----------- ImmutableSetTest.cpp - ImmutableSet unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ImmutableSet.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using namespace llvm;

namespace {
class ImmutableSetTest : public testing::Test {
protected:
  // for callback tests
  static char buffer[10];

  struct MyIter {
    int counter;
    char *ptr;

    MyIter() : counter(0), ptr(buffer) {
      for (unsigned i=0; i<sizeof(buffer);++i) buffer[i]='\0';
    }
    void operator()(char c) {
      *ptr++ = c;
      ++counter;
    }
  };
};
char ImmutableSetTest::buffer[10];


TEST_F(ImmutableSetTest, EmptyIntSetTest) {
  ImmutableSet<int>::Factory f;

  EXPECT_TRUE(f.getEmptySet() == f.getEmptySet());
  EXPECT_FALSE(f.getEmptySet() != f.getEmptySet());
  EXPECT_TRUE(f.getEmptySet().isEmpty());

  ImmutableSet<int> S = f.getEmptySet();
  EXPECT_EQ(0u, S.getHeight());
  EXPECT_TRUE(S.begin() == S.end());
  EXPECT_FALSE(S.begin() != S.end());
}


TEST_F(ImmutableSetTest, OneElemIntSetTest) {
  ImmutableSet<int>::Factory f;
  ImmutableSet<int> S = f.getEmptySet();

  ImmutableSet<int> S2 = f.add(S, 3);
  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S2.isEmpty());
  EXPECT_FALSE(S == S2);
  EXPECT_TRUE(S != S2);
  EXPECT_FALSE(S.contains(3));
  EXPECT_TRUE(S2.contains(3));
  EXPECT_FALSE(S2.begin() == S2.end());
  EXPECT_TRUE(S2.begin() != S2.end());

  ImmutableSet<int> S3 = f.add(S, 2);
  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S3.isEmpty());
  EXPECT_FALSE(S == S3);
  EXPECT_TRUE(S != S3);
  EXPECT_FALSE(S.contains(2));
  EXPECT_TRUE(S3.contains(2));

  EXPECT_FALSE(S2 == S3);
  EXPECT_TRUE(S2 != S3);
  EXPECT_FALSE(S2.contains(2));
  EXPECT_FALSE(S3.contains(3));
}

TEST_F(ImmutableSetTest, MultiElemIntSetTest) {
  ImmutableSet<int>::Factory f;
  ImmutableSet<int> S = f.getEmptySet();

  ImmutableSet<int> S2 = f.add(f.add(f.add(S, 3), 4), 5);
  ImmutableSet<int> S3 = f.add(f.add(f.add(S2, 9), 20), 43);
  ImmutableSet<int> S4 = f.add(S2, 9);

  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S2.isEmpty());
  EXPECT_FALSE(S3.isEmpty());
  EXPECT_FALSE(S4.isEmpty());

  EXPECT_FALSE(S.contains(3));
  EXPECT_FALSE(S.contains(9));

  EXPECT_TRUE(S2.contains(3));
  EXPECT_TRUE(S2.contains(4));
  EXPECT_TRUE(S2.contains(5));
  EXPECT_FALSE(S2.contains(9));
  EXPECT_FALSE(S2.contains(0));

  EXPECT_TRUE(S3.contains(43));
  EXPECT_TRUE(S3.contains(20));
  EXPECT_TRUE(S3.contains(9));
  EXPECT_TRUE(S3.contains(3));
  EXPECT_TRUE(S3.contains(4));
  EXPECT_TRUE(S3.contains(5));
  EXPECT_FALSE(S3.contains(0));

  EXPECT_TRUE(S4.contains(9));
  EXPECT_TRUE(S4.contains(3));
  EXPECT_TRUE(S4.contains(4));
  EXPECT_TRUE(S4.contains(5));
  EXPECT_FALSE(S4.contains(20));
  EXPECT_FALSE(S4.contains(43));
}

TEST_F(ImmutableSetTest, RemoveIntSetTest) {
  ImmutableSet<int>::Factory f;
  ImmutableSet<int> S = f.getEmptySet();

  ImmutableSet<int> S2 = f.add(f.add(S, 4), 5);
  ImmutableSet<int> S3 = f.add(S2, 3);
  ImmutableSet<int> S4 = f.remove(S3, 3);

  EXPECT_TRUE(S3.contains(3));
  EXPECT_FALSE(S2.contains(3));
  EXPECT_FALSE(S4.contains(3));

  EXPECT_TRUE(S2 == S4);
  EXPECT_TRUE(S3 != S2);
  EXPECT_TRUE(S3 != S4);

  EXPECT_TRUE(S3.contains(4));
  EXPECT_TRUE(S3.contains(5));

  EXPECT_TRUE(S4.contains(4));
  EXPECT_TRUE(S4.contains(5));
}

TEST_F(ImmutableSetTest, IterLongSetTest) {
  ImmutableSet<long>::Factory f;
  ImmutableSet<long> S = f.getEmptySet();

  ImmutableSet<long> S2 = f.add(f.add(f.add(S, 0), 1), 2);
  ImmutableSet<long> S3 = f.add(f.add(f.add(S2, 3), 4), 5);

  int i = 0;
  for (ImmutableSet<long>::iterator I = S.begin(), E = S.end(); I != E; ++I) {
    i++;
  }
  ASSERT_EQ(0, i);

  i = 0;
  for (ImmutableSet<long>::iterator I = S2.begin(), E = S2.end(); I != E; ++I) {
    ASSERT_EQ(i, *I);
    i++;
  }
  ASSERT_EQ(3, i);

  i = 0;
  for (ImmutableSet<long>::iterator I = S3.begin(), E = S3.end(); I != E; I++) {
    ASSERT_EQ(i, *I);
    i++;
  }
  ASSERT_EQ(6, i);
}

TEST_F(ImmutableSetTest, AddIfNotFoundTest) {
  ImmutableSet<long>::Factory f(/*canonicalize=*/false);
  ImmutableSet<long> S = f.getEmptySet();
  S = f.add(S, 1);
  S = f.add(S, 2);
  S = f.add(S, 3);

  ImmutableSet<long> T1 = f.add(S, 1);
  ImmutableSet<long> T2 = f.add(S, 2);
  ImmutableSet<long> T3 = f.add(S, 3);
  EXPECT_EQ(S.getRoot(), T1.getRoot());
  EXPECT_EQ(S.getRoot(), T2.getRoot());
  EXPECT_EQ(S.getRoot(), T3.getRoot());

  ImmutableSet<long> U = f.add(S, 4);
  EXPECT_NE(S.getRoot(), U.getRoot());
}

TEST_F(ImmutableSetTest, RemoveIfNotFoundTest) {
  ImmutableSet<long>::Factory f(/*canonicalize=*/false);
  ImmutableSet<long> S = f.getEmptySet();
  S = f.add(S, 1);
  S = f.add(S, 2);
  S = f.add(S, 3);

  ImmutableSet<long> T = f.remove(S, 4);
  EXPECT_EQ(S.getRoot(), T.getRoot());

  ImmutableSet<long> U = f.remove(S, 3);
  EXPECT_NE(S.getRoot(), U.getRoot());
}

//===----------------------------------------------------------------------===//
// In-order iterator correctness, validated against independent oracles.
//
// These checks do not assume any particular iterator implementation; they pin
// the externally observable contract (in-order ordering, reverse traversal,
// and skipSubTree). This is what the clang static analyzer and the tree
// canonicalization machinery rely on.
//===----------------------------------------------------------------------===//

namespace {
using Info = ImutContainerInfo<int>;
using Tree = ImutAVLTree<Info>;
using TreeIter = Tree::iterator; // ImutAVLTreeInOrderIterator

// Build an ImmutableSet from the given values (in the given insertion order),
// optionally removing some afterwards, so trees of varied shape are produced.
ImmutableSet<int> buildSet(ImmutableSet<int>::Factory &F, ArrayRef<int> ToAdd,
                           ArrayRef<int> ToRemove = {}) {
  ImmutableSet<int> S = F.getEmptySet();
  for (int V : ToAdd)
    S = F.add(S, V);
  for (int V : ToRemove)
    S = F.remove(S, V);
  return S;
}

// A representative collection of trees: degenerate, hand-picked small shapes,
// a large balanced one, and many pseudo-random insert/remove mixes.
std::vector<ImmutableSet<int>> makeTestSets(ImmutableSet<int>::Factory &F) {
  std::vector<ImmutableSet<int>> Sets;
  Sets.push_back(F.getEmptySet());
  Sets.push_back(buildSet(F, {42}));
  Sets.push_back(buildSet(F, {1, 2, 3}));
  Sets.push_back(buildSet(F, {3, 2, 1}));
  Sets.push_back(buildSet(F, {2, 1, 3}));

  std::vector<int> Sorted(200);
  std::iota(Sorted.begin(), Sorted.end(), 0);
  Sets.push_back(buildSet(F, Sorted));

  std::mt19937 Rng(12345);
  for (int Trial = 0; Trial < 25; ++Trial) {
    std::vector<int> Vals(150);
    std::iota(Vals.begin(), Vals.end(), 0);
    std::shuffle(Vals.begin(), Vals.end(), Rng);
    std::vector<int> Removals(Vals.begin(), Vals.begin() + (Trial % 40));
    Sets.push_back(buildSet(F, Vals, Removals));
  }
  return Sets;
}
} // namespace

// Forward iteration must visit keys in ascending order, exactly matching an
// independent std::set holding the same elements.
TEST_F(ImmutableSetTest, IteratorInOrderMatchesStdSet) {
  ImmutableSet<int>::Factory F(/*canonicalize=*/false);
  for (const ImmutableSet<int> &S : makeTestSets(F)) {
    std::set<int> Oracle;
    for (ImmutableSet<int>::iterator I = S.begin(), E = S.end(); I != E; ++I)
      Oracle.insert(*I);

    std::vector<int> Forward;
    for (ImmutableSet<int>::iterator I = S.begin(), E = S.end(); I != E; ++I)
      Forward.push_back(*I);

    EXPECT_TRUE(std::is_sorted(Forward.begin(), Forward.end()));
    EXPECT_TRUE(std::adjacent_find(Forward.begin(), Forward.end()) ==
                Forward.end()); // no duplicates
    EXPECT_TRUE(std::equal(Forward.begin(), Forward.end(), Oracle.begin(),
                           Oracle.end()));
  }
}

// Walking backwards with operator-- from the last element must reproduce the
// reverse of the forward traversal.
TEST_F(ImmutableSetTest, IteratorReverseMatchesForward) {
  ImmutableSet<int>::Factory F(/*canonicalize=*/false);
  for (const ImmutableSet<int> &S : makeTestSets(F)) {
    const Tree *Root = S.getRootWithoutRetain();

    std::vector<const Tree *> Forward;
    for (TreeIter I(Root), E; I != E; ++I)
      Forward.push_back(&*I);
    if (Forward.empty())
      continue;

    // Advance to the last element, then walk backwards.
    TreeIter Last(Root);
    for (TreeIter I(Root), E; I != E; ++I)
      Last = I;

    std::vector<const Tree *> Backward;
    Backward.push_back(&*Last);
    for (TreeIter B = Last; Backward.size() < Forward.size();) {
      --B;
      Backward.push_back(&*B);
    }
    std::reverse(Backward.begin(), Backward.end());
    EXPECT_EQ(Forward, Backward);
  }
}

// skipSubTree must land on the in-order successor of the *entire* subtree
// rooted at the current node. Since an in-order traversal visits a subtree as a
// contiguous run, the destination index is computable independently from the
// node's right-subtree size.
TEST_F(ImmutableSetTest, IteratorSkipSubTree) {
  ImmutableSet<int>::Factory F(/*canonicalize=*/false);
  for (const ImmutableSet<int> &S : makeTestSets(F)) {
    const Tree *Root = S.getRootWithoutRetain();

    std::vector<const Tree *> Order;
    for (TreeIter I(Root), E; I != E; ++I)
      Order.push_back(&*I);

    // From each starting index, skipSubTree should jump past the current node
    // and its whole right subtree.
    for (size_t Start = 0; Start < Order.size(); ++Start) {
      TreeIter I(Root);
      for (size_t N = 0; N < Start; ++N)
        ++I;
      const Tree *Node = &*I;
      unsigned RightSize = Node->getRight() ? Node->getRight()->size() : 0;
      size_t ExpectedIdx = Start + 1 + RightSize;

      I.skipSubTree();
      if (ExpectedIdx >= Order.size())
        EXPECT_TRUE(I == TreeIter()); // reached end
      else
        EXPECT_EQ(Order[ExpectedIdx], &*I);
    }
  }
}

// Structural equality (used by canonicalization) must agree with element-wise
// comparison of the in-order sequences. This exercises ++ and skipSubTree on
// the shared-subtree fast path inside ImutAVLTree::isEqual.
TEST_F(ImmutableSetTest, StructuralEqualityMatchesContents) {
  ImmutableSet<int>::Factory F(/*canonicalize=*/false);
  std::vector<ImmutableSet<int>> Sets = makeTestSets(F);
  for (const ImmutableSet<int> &A : Sets) {
    for (const ImmutableSet<int> &B : Sets) {
      std::vector<int> VA(A.begin(), A.end());
      std::vector<int> VB(B.begin(), B.end());
      EXPECT_EQ(VA == VB, A == B);
    }
  }
}

// Two independent iterators compare equal iff they sit on the same node, and
// equal-to-end iff both are at end. Exercises the node-identity operator==.
TEST_F(ImmutableSetTest, IteratorEquality) {
  ImmutableSet<int>::Factory F(/*canonicalize=*/false);
  for (const ImmutableSet<int> &S : makeTestSets(F)) {
    const Tree *Root = S.getRootWithoutRetain();
    size_t N = std::distance(TreeIter(Root), TreeIter());

    for (size_t I = 0; I < N; ++I) {
      TreeIter A(Root), B(Root);
      for (size_t K = 0; K < I; ++K) {
        ++A;
        ++B;
      }
      EXPECT_TRUE(A == B); // same position
      EXPECT_FALSE(A != B);
      EXPECT_TRUE(A != TreeIter()); // not end
      TreeIter C = B;
      ++C;
      EXPECT_TRUE(A != C); // adjacent positions differ
    }

    // Walking both to the end makes them equal to each other and to end().
    TreeIter A(Root), B(Root);
    while (A != TreeIter())
      ++A;
    while (B != TreeIter())
      ++B;
    EXPECT_TRUE(A == B);
    EXPECT_TRUE(A == TreeIter());
  }
}
} // namespace
