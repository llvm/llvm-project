//===- unittests/ADT/IListNodeTest.cpp - ilist_node unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist_node.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace llvm;
using namespace llvm::ilist_detail;

namespace {

struct Node;

struct TagA {};
struct TagB {};

TEST(IListNodeTest, Options) {
  static_assert(
      std::is_same_v<compute_node_options<Node>::type,
                     compute_node_options<Node, ilist_tag<void>>::type>,
      "default tag is void");
  static_assert(
      !std::is_same_v<compute_node_options<Node, ilist_tag<TagA>>::type,
                      compute_node_options<Node, ilist_tag<void>>::type>,
      "default tag is void, different from TagA");
  static_assert(
      !std::is_same_v<compute_node_options<Node, ilist_tag<TagA>>::type,
                      compute_node_options<Node, ilist_tag<TagB>>::type>,
      "TagA is not TagB");
  static_assert(
      std::is_same_v<
          compute_node_options<Node, ilist_sentinel_tracking<false>>::type,
          compute_node_options<Node, ilist_sentinel_tracking<false>,
                               ilist_tag<void>>::type>,
      "default tag is void, even with sentinel tracking off");
  static_assert(
      std::is_same_v<
          compute_node_options<Node, ilist_sentinel_tracking<false>>::type,
          compute_node_options<Node, ilist_tag<void>,
                               ilist_sentinel_tracking<false>>::type>,
      "order shouldn't matter");
  static_assert(
      std::is_same_v<
          compute_node_options<Node, ilist_sentinel_tracking<true>>::type,
          compute_node_options<Node, ilist_sentinel_tracking<true>,
                               ilist_tag<void>>::type>,
      "default tag is void, even with sentinel tracking on");
  static_assert(
      std::is_same_v<
          compute_node_options<Node, ilist_sentinel_tracking<true>>::type,
          compute_node_options<Node, ilist_tag<void>,
                               ilist_sentinel_tracking<true>>::type>,
      "order shouldn't matter");
  static_assert(
      std::is_same_v<compute_node_options<Node, ilist_sentinel_tracking<true>,
                                          ilist_tag<TagA>>::type,
                     compute_node_options<Node, ilist_tag<TagA>,
                                          ilist_sentinel_tracking<true>>::type>,
      "order shouldn't matter with real tags");
}

} // end namespace
