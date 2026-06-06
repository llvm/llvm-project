//===- BPFunctionNodeTest.cpp - BPFunctionNode tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/BalancedPartitioning.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::Field;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

namespace llvm {

void PrintTo(const BPFunctionNode &Node, std::ostream *OS) {
  raw_os_ostream ROS(*OS);
  Node.dump(ROS);
}

TEST(BPFunctionNodeTest, Basic) {
  auto NodeIs = [](BPFunctionNode::IDT Id,
                   ArrayRef<BPFunctionNode::UtilityNodeT> UNs) {
    return AllOf(Field("Id", &BPFunctionNode::Id, Id),
                 Field("UtilityNodes", &BPFunctionNode::UtilityNodes,
                       UnorderedElementsAreArray(UNs)));
  };

  std::vector<BPFunctionNode> Nodes;
  TemporalProfTraceTy::createBPFunctionNodes(
      {TemporalProfTraceTy({0, 1, 2, 3})}, Nodes, /*RemoveOutlierUNs=*/false);
  // Utility nodes that are too infrequent or too prevalent are filtered out.
  EXPECT_THAT(Nodes,
              UnorderedElementsAre(NodeIs(0, {0, 1, 2}), NodeIs(1, {1, 2}),
                                   NodeIs(2, {2}), NodeIs(3, {2})));

  Nodes.clear();
  TemporalProfTraceTy::createBPFunctionNodes(
      {TemporalProfTraceTy({0, 1, 2, 3, 4}), TemporalProfTraceTy({4, 2})},
      Nodes, /*RemoveOutlierUNs=*/false);

  EXPECT_THAT(Nodes,
              UnorderedElementsAre(NodeIs(0, {0, 1, 2, 3}),
                                   NodeIs(1, {1, 2, 3}), NodeIs(2, {2, 3, 5}),
                                   NodeIs(3, {2, 3}), NodeIs(4, {3, 4, 5})));

  Nodes.clear();
  TemporalProfTraceTy::createBPFunctionNodes(
      {TemporalProfTraceTy({0, 1, 2, 3, 4}), TemporalProfTraceTy({4, 2})},
      Nodes, /*RemoveOutlierUNs=*/true);

  EXPECT_THAT(Nodes, UnorderedElementsAre(NodeIs(0, {1}), NodeIs(1, {1}),
                                          NodeIs(2, {5}), NodeIs(3, {}),
                                          NodeIs(4, {5})));
}

} // end namespace llvm
