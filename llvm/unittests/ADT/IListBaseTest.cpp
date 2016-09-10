//===- unittests/ADT/IListBaseTest.cpp - ilist_base unit tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist_base.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

typedef ilist_base list_base_type;
typedef ilist_node_base node_base_type;

TEST(IListBaseTest, insertBeforeImpl) {
  node_base_type S, A, B;
  // [S] <-> [S]
  S.setPrev(&S);
  S.setNext(&S);

  // [S] <-> A <-> [S]
  list_base_type::insertBeforeImpl(S, A);
  EXPECT_EQ(&A, S.getPrev());
  EXPECT_EQ(&S, A.getPrev());
  EXPECT_EQ(&A, S.getNext());
  EXPECT_EQ(&S, A.getNext());

  // [S] <-> A <-> B <-> [S]
  list_base_type::insertBeforeImpl(S, B);
  EXPECT_EQ(&B, S.getPrev());
  EXPECT_EQ(&A, B.getPrev());
  EXPECT_EQ(&S, A.getPrev());
  EXPECT_EQ(&A, S.getNext());
  EXPECT_EQ(&B, A.getNext());
  EXPECT_EQ(&S, B.getNext());
}

TEST(IListBaseTest, removeImpl) {
  node_base_type S, A, B;

  // [S] <-> A <-> B <-> [S]
  S.setPrev(&S);
  S.setNext(&S);
  list_base_type::insertBeforeImpl(S, A);
  list_base_type::insertBeforeImpl(S, B);

  // [S] <-> B <-> [S]
  list_base_type::removeImpl(A);
  EXPECT_EQ(&B, S.getPrev());
  EXPECT_EQ(&S, B.getPrev());
  EXPECT_EQ(&B, S.getNext());
  EXPECT_EQ(&S, B.getNext());
  EXPECT_EQ(nullptr, A.getPrev());
  EXPECT_EQ(nullptr, A.getNext());

  // [S] <-> [S]
  list_base_type::removeImpl(B);
  EXPECT_EQ(&S, S.getPrev());
  EXPECT_EQ(&S, S.getNext());
  EXPECT_EQ(nullptr, B.getPrev());
  EXPECT_EQ(nullptr, B.getNext());
}

TEST(IListBaseTest, removeRangeImpl) {
  node_base_type S, A, B, C, D;

  // [S] <-> A <-> B <-> C <-> D <-> [S]
  S.setPrev(&S);
  S.setNext(&S);
  list_base_type::insertBeforeImpl(S, A);
  list_base_type::insertBeforeImpl(S, B);
  list_base_type::insertBeforeImpl(S, C);
  list_base_type::insertBeforeImpl(S, D);

  // [S] <-> A <-> D <-> [S]
  list_base_type::removeRangeImpl(B, D);
  EXPECT_EQ(&D, S.getPrev());
  EXPECT_EQ(&A, D.getPrev());
  EXPECT_EQ(&S, A.getPrev());
  EXPECT_EQ(&A, S.getNext());
  EXPECT_EQ(&D, A.getNext());
  EXPECT_EQ(&S, D.getNext());
  EXPECT_EQ(nullptr, B.getPrev());
  EXPECT_EQ(nullptr, C.getNext());
}

TEST(IListBaseTest, removeRangeImplAllButSentinel) {
  node_base_type S, A, B;

  // [S] <-> A <-> B <-> [S]
  S.setPrev(&S);
  S.setNext(&S);
  list_base_type::insertBeforeImpl(S, A);
  list_base_type::insertBeforeImpl(S, B);

  // [S] <-> [S]
  list_base_type::removeRangeImpl(A, S);
  EXPECT_EQ(&S, S.getPrev());
  EXPECT_EQ(&S, S.getNext());
  EXPECT_EQ(nullptr, A.getPrev());
  EXPECT_EQ(nullptr, B.getNext());
}

TEST(IListBaseTest, transferBeforeImpl) {
  node_base_type S1, S2, A, B, C, D, E;

  // [S1] <-> A <-> B <-> C <-> [S1]
  S1.setPrev(&S1);
  S1.setNext(&S1);
  list_base_type::insertBeforeImpl(S1, A);
  list_base_type::insertBeforeImpl(S1, B);
  list_base_type::insertBeforeImpl(S1, C);

  // [S2] <-> D <-> E <-> [S2]
  S2.setPrev(&S2);
  S2.setNext(&S2);
  list_base_type::insertBeforeImpl(S2, D);
  list_base_type::insertBeforeImpl(S2, E);

  // [S1] <-> C <-> [S1]
  list_base_type::transferBeforeImpl(D, A, C);
  EXPECT_EQ(&C, S1.getPrev());
  EXPECT_EQ(&S1, C.getPrev());
  EXPECT_EQ(&C, S1.getNext());
  EXPECT_EQ(&S1, C.getNext());

  // [S2] <-> A <-> B <-> D <-> E <-> [S2]
  EXPECT_EQ(&E, S2.getPrev());
  EXPECT_EQ(&D, E.getPrev());
  EXPECT_EQ(&B, D.getPrev());
  EXPECT_EQ(&A, B.getPrev());
  EXPECT_EQ(&S2, A.getPrev());
  EXPECT_EQ(&A, S2.getNext());
  EXPECT_EQ(&B, A.getNext());
  EXPECT_EQ(&D, B.getNext());
  EXPECT_EQ(&E, D.getNext());
  EXPECT_EQ(&S2, E.getNext());
}

} // end namespace
