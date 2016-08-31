//===- unittests/ADT/SimpleIListTest.cpp - simple_ilist unit tests --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/simple_ilist.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct Node : ilist_node<Node> {};
bool operator<(const Node &L, const Node &R) { return &L < &R; }
bool makeFalse(const Node &, const Node &) { return false; }

struct deleteNode : std::default_delete<Node> {};
void doNothing(Node *) {}

TEST(SimpleIListTest, DefaultConstructor) {
  simple_ilist<Node> L;
  EXPECT_EQ(L.begin(), L.end());
  EXPECT_TRUE(L.empty());
  EXPECT_EQ(0u, L.size());
}

TEST(SimpleIListTest, pushPopFront) {
  simple_ilist<Node> L;
  Node A, B;
  L.push_front(B);
  L.push_front(A);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&B, &L.back());
  EXPECT_FALSE(L.empty());
  EXPECT_EQ(2u, L.size());

  // Pop front and check the new front.
  L.pop_front();
  EXPECT_EQ(&B, &L.front());

  // Pop to empty.
  L.pop_front();
  EXPECT_TRUE(L.empty());
}

TEST(SimpleIListTest, pushPopBack) {
  simple_ilist<Node> L;
  Node A, B;
  L.push_back(A);
  L.push_back(B);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&B, &L.back());
  EXPECT_FALSE(L.empty());
  EXPECT_EQ(2u, L.size());

  // Pop back and check the new front.
  L.pop_back();
  EXPECT_EQ(&A, &L.back());

  // Pop to empty.
  L.pop_back();
  EXPECT_TRUE(L.empty());
}

TEST(SimpleIListTest, swap) {
  simple_ilist<Node> L1, L2;
  Node A, B;
  L1.push_back(A);
  L1.push_back(B);
  L1.swap(L2);
  EXPECT_TRUE(L1.empty());
  EXPECT_EQ(0u, L1.size());
  EXPECT_EQ(&A, &L2.front());
  EXPECT_EQ(&B, &L2.back());
  EXPECT_FALSE(L2.empty());
  EXPECT_EQ(2u, L2.size());
}

TEST(SimpleIListTest, insertEraseAtEnd) {
  simple_ilist<Node> L;
  Node A, B;
  L.insert(L.end(), A);
  L.insert(L.end(), B);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&B, &L.back());
  EXPECT_FALSE(L.empty());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, insertAtBegin) {
  simple_ilist<Node> L;
  Node A, B;
  L.insert(L.begin(), B);
  L.insert(L.begin(), A);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&B, &L.back());
  EXPECT_FALSE(L.empty());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, remove) {
  simple_ilist<Node> L;
  Node A, B, C;
  L.push_back(A);
  L.push_back(B);
  L.push_back(C);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&B, &*++L.begin());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(3u, L.size());

  L.remove(B);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(2u, L.size());

  L.remove(A);
  EXPECT_EQ(&C, &L.front());
  EXPECT_EQ(1u, L.size());

  L.remove(C);
  EXPECT_TRUE(L.empty());
}

TEST(SimpleIListTest, removeAndDispose) {
  simple_ilist<Node> L;
  Node A, C;
  Node *B = new Node;
  L.push_back(A);
  L.push_back(*B);
  L.push_back(C);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(B, &*++L.begin());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(3u, L.size());

  L.removeAndDispose(*B, deleteNode());
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, removeAndDisposeNullDeleter) {
  simple_ilist<Node> L;
  Node A, B, C;
  L.push_back(A);
  L.push_back(B);
  L.push_back(C);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&B, &*++L.begin());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(3u, L.size());

  L.removeAndDispose(B, doNothing);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, erase) {
  simple_ilist<Node> L;
  Node A, B, C;
  L.push_back(A);
  L.push_back(B);
  L.push_back(C);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&B, &*++L.begin());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(3u, L.size());

  EXPECT_EQ(C.getIterator(), L.erase(B.getIterator()));
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, eraseAndDispose) {
  simple_ilist<Node> L;
  Node A, C;
  Node *B = new Node;
  L.push_back(A);
  L.push_back(*B);
  L.push_back(C);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(B, &*++L.begin());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(3u, L.size());

  L.eraseAndDispose(B->getIterator(), deleteNode());
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, eraseAndDisposeNullDeleter) {
  simple_ilist<Node> L;
  Node A, B, C;
  L.push_back(A);
  L.push_back(B);
  L.push_back(C);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&B, &*++L.begin());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(3u, L.size());

  L.eraseAndDispose(B.getIterator(), doNothing);
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&C, &L.back());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, eraseRange) {
  simple_ilist<Node> L;
  Node A, B, C, D, E;
  L.push_back(A);
  L.push_back(B);
  L.push_back(C);
  L.push_back(D);
  L.push_back(E);
  auto I = L.begin();
  EXPECT_EQ(&A, &*I++);
  EXPECT_EQ(&B, &*I++);
  EXPECT_EQ(&C, &*I++);
  EXPECT_EQ(&D, &*I++);
  EXPECT_EQ(&E, &*I++);
  EXPECT_EQ(L.end(), I);
  EXPECT_EQ(5u, L.size());

  // Erase a range.
  EXPECT_EQ(E.getIterator(), L.erase(B.getIterator(), E.getIterator()));
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&E, &L.back());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, eraseAndDisposeRange) {
  simple_ilist<Node> L;
  Node A, *B = new Node, *C = new Node, *D = new Node, E;
  L.push_back(A);
  L.push_back(*B);
  L.push_back(*C);
  L.push_back(*D);
  L.push_back(E);
  auto I = L.begin();
  EXPECT_EQ(&A, &*I++);
  EXPECT_EQ(B, &*I++);
  EXPECT_EQ(C, &*I++);
  EXPECT_EQ(D, &*I++);
  EXPECT_EQ(&E, &*I++);
  EXPECT_EQ(L.end(), I);
  EXPECT_EQ(5u, L.size());

  // Erase a range.
  EXPECT_EQ(E.getIterator(),
            L.eraseAndDispose(B->getIterator(), E.getIterator(), deleteNode()));
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&E, &L.back());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, eraseAndDisposeRangeNullDeleter) {
  simple_ilist<Node> L;
  Node A, B, C, D, E;
  L.push_back(A);
  L.push_back(B);
  L.push_back(C);
  L.push_back(D);
  L.push_back(E);
  auto I = L.begin();
  EXPECT_EQ(&A, &*I++);
  EXPECT_EQ(&B, &*I++);
  EXPECT_EQ(&C, &*I++);
  EXPECT_EQ(&D, &*I++);
  EXPECT_EQ(&E, &*I++);
  EXPECT_EQ(L.end(), I);
  EXPECT_EQ(5u, L.size());

  // Erase a range.
  EXPECT_EQ(E.getIterator(),
            L.eraseAndDispose(B.getIterator(), E.getIterator(), doNothing));
  EXPECT_EQ(&A, &L.front());
  EXPECT_EQ(&E, &L.back());
  EXPECT_EQ(2u, L.size());
}

TEST(SimpleIListTest, clear) {
  simple_ilist<Node> L;
  Node A, B;
  L.push_back(A);
  L.push_back(B);
  L.clear();
  EXPECT_TRUE(L.empty());
  EXPECT_EQ(0u, L.size());
}

TEST(SimpleIListTest, clearAndDispose) {
  simple_ilist<Node> L;
  Node *A = new Node;
  Node *B = new Node;
  L.push_back(*A);
  L.push_back(*B);
  L.clearAndDispose(deleteNode());
  EXPECT_TRUE(L.empty());
  EXPECT_EQ(0u, L.size());
}

TEST(SimpleIListTest, clearAndDisposeNullDeleter) {
  simple_ilist<Node> L;
  Node A, B;
  L.push_back(A);
  L.push_back(B);
  L.clearAndDispose(doNothing);
  EXPECT_TRUE(L.empty());
  EXPECT_EQ(0u, L.size());
}

TEST(SimpleIListTest, spliceList) {
  simple_ilist<Node> L1, L2;
  Node A, B, C, D;

  // [A, D].
  L1.push_back(A);
  L1.push_back(D);

  // [B, C].
  L2.push_back(B);
  L2.push_back(C);

  // Splice in L2, giving [A, B, C, D].
  L1.splice(--L1.end(), L2);
  EXPECT_TRUE(L2.empty());
  EXPECT_EQ(4u, L1.size());
  auto I = L1.begin();
  EXPECT_EQ(&A, &*I++);
  EXPECT_EQ(&B, &*I++);
  EXPECT_EQ(&C, &*I++);
  EXPECT_EQ(&D, &*I++);
  EXPECT_EQ(L1.end(), I);
}

TEST(SimpleIListTest, spliceSingle) {
  simple_ilist<Node> L1, L2;
  Node A, B, C, D, E;

  // [A, C].
  L1.push_back(A);
  L1.push_back(C);

  // [D, B, E].
  L2.push_back(D);
  L2.push_back(B);
  L2.push_back(E);

  // Splice B from L2 to L1, giving [A, B, C] and [D, E].
  L1.splice(--L1.end(), L2, ++L2.begin());
  auto I = L1.begin();
  EXPECT_EQ(&A, &*I++);
  EXPECT_EQ(&B, &*I++);
  EXPECT_EQ(&C, &*I++);
  EXPECT_EQ(L1.end(), I);

  I = L2.begin();
  EXPECT_EQ(&D, &*I++);
  EXPECT_EQ(&E, &*I++);
  EXPECT_EQ(L2.end(), I);
}

TEST(SimpleIListTest, spliceRange) {
  simple_ilist<Node> L1, L2;
  Node A, B, C, D, E, F;

  // [A, D].
  L1.push_back(A);
  L1.push_back(D);

  // [E, B, C, F].
  L2.push_back(E);
  L2.push_back(B);
  L2.push_back(C);
  L2.push_back(F);

  // Splice B from L2 to L1, giving [A, B, C, D] and [E, F].
  L1.splice(--L1.end(), L2, ++L2.begin(), --L2.end());
  auto I = L1.begin();
  EXPECT_EQ(&A, &*I++);
  EXPECT_EQ(&B, &*I++);
  EXPECT_EQ(&C, &*I++);
  EXPECT_EQ(&D, &*I++);
  EXPECT_EQ(L1.end(), I);

  I = L2.begin();
  EXPECT_EQ(&E, &*I++);
  EXPECT_EQ(&F, &*I++);
  EXPECT_EQ(L2.end(), I);
}

TEST(SimpleIListTest, merge) {
  for (bool IsL1LHS : {false, true}) {
    simple_ilist<Node> L1, L2;
    Node Ns[10];

    // Fill L1.
    L1.push_back(Ns[0]);
    L1.push_back(Ns[3]);
    L1.push_back(Ns[4]);
    L1.push_back(Ns[8]);

    // Fill L2.
    L2.push_back(Ns[1]);
    L2.push_back(Ns[2]);
    L2.push_back(Ns[5]);
    L2.push_back(Ns[6]);
    L2.push_back(Ns[7]);
    L2.push_back(Ns[9]);

    // Check setup.
    EXPECT_EQ(4u, L1.size());
    EXPECT_EQ(6u, L2.size());
    EXPECT_TRUE(std::is_sorted(L1.begin(), L1.end()));
    EXPECT_TRUE(std::is_sorted(L2.begin(), L2.end()));

    // Merge.
    auto &LHS = IsL1LHS ? L1 : L2;
    auto &RHS = IsL1LHS ? L2 : L1;
    LHS.merge(RHS);
    EXPECT_TRUE(RHS.empty());
    EXPECT_FALSE(LHS.empty());
    EXPECT_TRUE(std::is_sorted(LHS.begin(), LHS.end()));
    auto I = LHS.begin();
    for (Node &N : Ns)
      EXPECT_EQ(&N, &*I++);
    EXPECT_EQ(LHS.end(), I);
  }
}

TEST(SimpleIListTest, mergeIsStable) {
  simple_ilist<Node> L1, L2;
  Node Ns[5];

  auto setup = [&]() {
    EXPECT_TRUE(L1.empty());
    EXPECT_TRUE(L2.empty());

    // Fill L1.
    L1.push_back(Ns[0]);
    L1.push_back(Ns[3]);
    L1.push_back(Ns[4]);

    // Fill L2.
    L2.push_back(Ns[1]);
    L2.push_back(Ns[2]);

    // Check setup.
    EXPECT_EQ(3u, L1.size());
    EXPECT_EQ(2u, L2.size());
    EXPECT_TRUE(std::is_sorted(L1.begin(), L1.end(), makeFalse));
    EXPECT_TRUE(std::is_sorted(L2.begin(), L2.end(), makeFalse));
  };

  // Merge.  Should be stable.
  setup();
  L1.merge(L2, makeFalse);
  EXPECT_TRUE(L2.empty());
  EXPECT_FALSE(L1.empty());
  EXPECT_TRUE(std::is_sorted(L1.begin(), L1.end(), makeFalse));
  auto I = L1.begin();
  EXPECT_EQ(&Ns[0], &*I++);
  EXPECT_EQ(&Ns[3], &*I++);
  EXPECT_EQ(&Ns[4], &*I++);
  EXPECT_EQ(&Ns[1], &*I++);
  EXPECT_EQ(&Ns[2], &*I++);
  EXPECT_EQ(L1.end(), I);

  // Merge the other way.  Should be stable.
  L1.clear();
  setup();
  L2.merge(L1, makeFalse);
  EXPECT_TRUE(L1.empty());
  EXPECT_FALSE(L2.empty());
  EXPECT_TRUE(std::is_sorted(L2.begin(), L2.end(), makeFalse));
  I = L2.begin();
  EXPECT_EQ(&Ns[1], &*I++);
  EXPECT_EQ(&Ns[2], &*I++);
  EXPECT_EQ(&Ns[0], &*I++);
  EXPECT_EQ(&Ns[3], &*I++);
  EXPECT_EQ(&Ns[4], &*I++);
  EXPECT_EQ(L2.end(), I);
}

TEST(SimpleIListTest, mergeEmpty) {
  for (bool IsL1LHS : {false, true}) {
    simple_ilist<Node> L1, L2;
    Node Ns[4];

    // Fill L1.
    L1.push_back(Ns[0]);
    L1.push_back(Ns[1]);
    L1.push_back(Ns[2]);
    L1.push_back(Ns[3]);

    // Check setup.
    EXPECT_EQ(4u, L1.size());
    EXPECT_TRUE(L2.empty());
    EXPECT_TRUE(std::is_sorted(L1.begin(), L1.end()));

    // Merge.
    auto &LHS = IsL1LHS ? L1 : L2;
    auto &RHS = IsL1LHS ? L2 : L1;
    LHS.merge(RHS);
    EXPECT_TRUE(RHS.empty());
    EXPECT_FALSE(LHS.empty());
    EXPECT_TRUE(std::is_sorted(LHS.begin(), LHS.end()));
    auto I = LHS.begin();
    for (Node &N : Ns)
      EXPECT_EQ(&N, &*I++);
    EXPECT_EQ(LHS.end(), I);
  }
}

TEST(SimpleIListTest, mergeBothEmpty) {
  simple_ilist<Node> L1, L2;
  L1.merge(L2);
  EXPECT_TRUE(L1.empty());
  EXPECT_TRUE(L2.empty());
}

TEST(SimpleIListTest, sort) {
  simple_ilist<Node> L;
  Node Ns[10];

  // Fill L.
  for (int I : {3, 4, 0, 8, 1, 2, 6, 7, 9, 5})
    L.push_back(Ns[I]);

  // Check setup.
  EXPECT_EQ(10u, L.size());
  EXPECT_FALSE(std::is_sorted(L.begin(), L.end()));

  // Sort.
  L.sort();
  EXPECT_TRUE(std::is_sorted(L.begin(), L.end()));
  auto I = L.begin();
  for (Node &N : Ns)
    EXPECT_EQ(&N, &*I++);
  EXPECT_EQ(L.end(), I);
}

TEST(SimpleIListTest, sortIsStable) {
  simple_ilist<Node> L;
  Node Ns[10];

  // Compare such that nodes are partitioned but not fully sorted.
  auto partition = [&](const Node &N) { return &N >= &Ns[5]; };
  auto compare = [&](const Node &L, const Node &R) {
    return partition(L) < partition(R);
  };

  // Fill L.
  for (int I : {3, 4, 7, 8, 1, 2, 6, 0, 9, 5})
    L.push_back(Ns[I]);

  // Check setup.
  EXPECT_EQ(10u, L.size());
  EXPECT_FALSE(std::is_sorted(L.begin(), L.end(), compare));

  // Sort.
  L.sort(compare);
  EXPECT_TRUE(std::is_sorted(L.begin(), L.end(), compare));
  auto I = L.begin();
  for (int O : {3, 4, 1, 2, 0})
    EXPECT_EQ(&Ns[O], &*I++);
  for (int O : {7, 8, 6, 9, 5})
    EXPECT_EQ(&Ns[O], &*I++);
  EXPECT_EQ(L.end(), I);
}

TEST(SimpleIListTest, sortEmpty) {
  simple_ilist<Node> L;
  L.sort();
}

} // end namespace
