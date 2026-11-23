//===- unittests/ADT/IListSentinelTest.cpp - ilist_sentinel unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist_node.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <class T, class... Options> struct PickSentinel {
  using type = ilist_sentinel<
      typename ilist_detail::compute_node_options<T, Options...>::type>;
};

class Node : public ilist_node<Node> {};
class TrackingNode : public ilist_node<Node, ilist_sentinel_tracking<true>> {};
using Sentinel = PickSentinel<Node>::type;
using TrackingSentinel =
    PickSentinel<Node, ilist_sentinel_tracking<true>>::type;
using NoTrackingSentinel =
    PickSentinel<Node, ilist_sentinel_tracking<false>>::type;

struct LocalAccess : ilist_detail::NodeAccess {
  using NodeAccess::getPrev;
  using NodeAccess::getNext;
};

TEST(IListSentinelTest, DefaultConstructor) {
  Sentinel S;
  EXPECT_EQ(&S, LocalAccess::getPrev(S));
  EXPECT_EQ(&S, LocalAccess::getNext(S));
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  EXPECT_TRUE(S.isKnownSentinel());
#else
  EXPECT_FALSE(S.isKnownSentinel());
#endif

  TrackingSentinel TS;
  NoTrackingSentinel NTS;
  EXPECT_TRUE(TS.isSentinel());
  EXPECT_TRUE(TS.isKnownSentinel());
  EXPECT_FALSE(NTS.isKnownSentinel());
}

TEST(IListSentinelTest, NormalNodeIsNotKnownSentinel) {
  Node N;
  EXPECT_EQ(nullptr, LocalAccess::getPrev(N));
  EXPECT_EQ(nullptr, LocalAccess::getNext(N));
  EXPECT_FALSE(N.isKnownSentinel());

  TrackingNode TN;
  EXPECT_FALSE(TN.isSentinel());
}

} // end namespace
