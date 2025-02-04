//===- unittests/ADT/IListBaseTest.cpp - ilist_base unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist_base.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Test fixture.
template <typename T> class IListBaseTest : public ::testing::Test {};

class Parent;

// Test variants with the same test.
typedef ::testing::Types<ilist_base<false, void>, ilist_base<true, void>, ilist_base<false, Parent*>, ilist_base<true, Parent*>>
    IListBaseTestTypes;
TYPED_TEST_SUITE(IListBaseTest, IListBaseTestTypes, );

TYPED_TEST(IListBaseTest, insertBeforeImpl) {
  typedef TypeParam list_base_type;
  typedef typename list_base_type::node_base_type node_base_type;

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

TYPED_TEST(IListBaseTest, removeImpl) {
  typedef TypeParam list_base_type;
  typedef typename list_base_type::node_base_type node_base_type;

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

TYPED_TEST(IListBaseTest, removeRangeImpl) {
  typedef TypeParam list_base_type;
  typedef typename list_base_type::node_base_type node_base_type;

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

TYPED_TEST(IListBaseTest, removeRangeImplAllButSentinel) {
  typedef TypeParam list_base_type;
  typedef typename list_base_type::node_base_type node_base_type;

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

TYPED_TEST(IListBaseTest, transferBeforeImpl) {
  typedef TypeParam list_base_type;
  typedef typename list_base_type::node_base_type node_base_type;

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
