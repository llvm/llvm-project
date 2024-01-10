//==- unittests/ADT/IListIteratorBitsTest.cpp - ilist_iterator_w_bits tests -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/simple_ilist.h"
#include "gtest/gtest.h"

// Test that ilist_iterator_w_bits can be used to store extra information about
// what we're iterating over, that it's only enabled when given the relevant
// option, and it can be fed into various iteration utilities.

using namespace llvm;

namespace {

class dummy;

struct Node : ilist_node<Node, ilist_iterator_bits<true>> {
  friend class dummy;
};

struct PlainNode : ilist_node<PlainNode> {
  friend class dummy;
};

TEST(IListIteratorBitsTest, DefaultConstructor) {
  simple_ilist<Node, ilist_iterator_bits<true>>::iterator I;
  simple_ilist<Node, ilist_iterator_bits<true>>::reverse_iterator RI;
  simple_ilist<Node, ilist_iterator_bits<true>>::const_iterator CI;
  simple_ilist<Node, ilist_iterator_bits<true>>::const_reverse_iterator CRI;
  EXPECT_EQ(nullptr, I.getNodePtr());
  EXPECT_EQ(nullptr, CI.getNodePtr());
  EXPECT_EQ(nullptr, RI.getNodePtr());
  EXPECT_EQ(nullptr, CRI.getNodePtr());
  EXPECT_EQ(I, I);
  EXPECT_EQ(I, CI);
  EXPECT_EQ(CI, I);
  EXPECT_EQ(CI, CI);
  EXPECT_EQ(RI, RI);
  EXPECT_EQ(RI, CRI);
  EXPECT_EQ(CRI, RI);
  EXPECT_EQ(CRI, CRI);
  EXPECT_EQ(I, RI.getReverse());
  EXPECT_EQ(RI, I.getReverse());
}

TEST(IListIteratorBitsTest, ConsAndAssignment) {
  simple_ilist<Node, ilist_iterator_bits<true>> L;
  Node A;
  L.insert(L.end(), A);

  simple_ilist<Node, ilist_iterator_bits<true>>::iterator I, I2;

// Two sets of tests: if we've compiled in the iterator bits, then check that
// HeadInclusiveBit and TailInclusiveBit are preserved on assignment and copy
// construction, but not on other operations.
#ifdef EXPERIMENTAL_DEBUGINFO_ITERATORS
  I = L.begin();
  EXPECT_FALSE(I.getHeadBit());
  EXPECT_FALSE(I.getTailBit());
  I.setHeadBit(true);
  I.setTailBit(true);
  EXPECT_TRUE(I.getHeadBit());
  EXPECT_TRUE(I.getTailBit());

  ++I;

  EXPECT_FALSE(I.getHeadBit());
  EXPECT_FALSE(I.getTailBit());

  I = L.begin();
  I.setHeadBit(true);
  I.setTailBit(true);
  I2 = I;
  EXPECT_TRUE(I2.getHeadBit());
  EXPECT_TRUE(I2.getTailBit());

  I = L.begin();
  I.setHeadBit(true);
  I.setTailBit(true);
  simple_ilist<Node, ilist_iterator_bits<true>>::iterator I3(I);
  EXPECT_TRUE(I3.getHeadBit());
  EXPECT_TRUE(I3.getTailBit());
#else
  // The calls should be available, but shouldn't actually store information.
  I = L.begin();
  EXPECT_FALSE(I.getHeadBit());
  EXPECT_FALSE(I.getTailBit());
  I.setHeadBit(true);
  I.setTailBit(true);
  EXPECT_FALSE(I.getHeadBit());
  EXPECT_FALSE(I.getTailBit());
  // Suppress warnings as we don't test with this variable.
  (void)I2;
#endif
}

class dummy {
  // Test that we get an ilist_iterator_w_bits out of the node given that the
  // options are enabled.
  using node_options = typename ilist_detail::compute_node_options<
      Node, ilist_iterator_bits<true>>::type;
  static_assert(std::is_same<Node::self_iterator,
                             llvm::ilist_iterator_w_bits<node_options, false,
                                                         false>>::value);

  // Now test that a plain node, without the option, gets a plain
  // ilist_iterator.
  using plain_node_options =
      typename ilist_detail::compute_node_options<PlainNode>::type;
  static_assert(std::is_same<
                PlainNode::self_iterator,
                llvm::ilist_iterator<plain_node_options, false, false>>::value);
};

TEST(IListIteratorBitsTest, RangeIteration) {
  // Check that we can feed ilist_iterator_w_bits into make_range and similar.
  // Plus, we should be able to convert it to a reverse iterator and use that.
  simple_ilist<Node, ilist_iterator_bits<true>> L;
  Node A;
  L.insert(L.end(), A);

  for (Node &N : make_range(L.begin(), L.end()))
    (void)N;

  simple_ilist<Node, ilist_iterator_bits<true>>::iterator It =
      L.begin()->getIterator();
  auto RevIt = It.getReverse();

  for (Node &N : make_range(RevIt, L.rend()))
    (void)N;
}

} // end namespace
