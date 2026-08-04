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
                   ArrayRef<BPFunctionNode::WeightedUtilityNode> UNs) {
    return AllOf(Field("Id", &BPFunctionNode::Id, Id),
                 Field("UtilityNodes", &BPFunctionNode::UtilityNodes,
                       UnorderedElementsAreArray(UNs)));
  };

  std::vector<BPFunctionNode> Nodes;
  TemporalProfTraceTy::createBPFunctionNodes(
      {TemporalProfTraceTy({0, 1, 2, 3})}, Nodes, /*RemoveOutlierUNs=*/false);
  // Utility nodes that are too infrequent or too prevalent are filtered out.
  EXPECT_THAT(Nodes,
              UnorderedElementsAre(NodeIs(0, {{0, 1}, {1, 1}, {2, 1}}),
                                   NodeIs(1, {{1, 1}, {2, 1}}),
                                   NodeIs(2, {{2, 1}}), NodeIs(3, {{2, 1}})));

  Nodes.clear();
  TemporalProfTraceTy::createBPFunctionNodes(
      {TemporalProfTraceTy({0, 1, 2, 3, 4}), TemporalProfTraceTy({4, 2})},
      Nodes, /*RemoveOutlierUNs=*/false);

  EXPECT_THAT(Nodes,
              UnorderedElementsAre(NodeIs(0, {{0, 1}, {1, 1}, {2, 1}, {3, 1}}),
                                   NodeIs(1, {{1, 1}, {2, 1}, {3, 1}}),
                                   NodeIs(2, {{2, 1}, {3, 1}, {5, 1}}),
                                   NodeIs(3, {{2, 1}, {3, 1}}),
                                   NodeIs(4, {{3, 1}, {4, 1}, {5, 1}})));

  Nodes.clear();
  TemporalProfTraceTy::createBPFunctionNodes(
      {TemporalProfTraceTy({0, 1, 2, 3, 4}), TemporalProfTraceTy({4, 2})},
      Nodes, /*RemoveOutlierUNs=*/true);

  EXPECT_THAT(Nodes,
              UnorderedElementsAre(NodeIs(0, {{1, 1}}), NodeIs(1, {{1, 1}}),
                                   NodeIs(2, {{5, 1}}), NodeIs(3, {}),
                                   NodeIs(4, {{5, 1}})));

  Nodes.clear();
  TemporalProfTraceTy::createBPFunctionNodes(
      {TemporalProfTraceTy({0, 1, 2, 3}, 10), TemporalProfTraceTy({0, 4, 1}, 1),
       TemporalProfTraceTy({5, 6}, 0)},
      Nodes, /*RemoveOutlierUNs=*/false);
  EXPECT_THAT(
      Nodes, UnorderedElementsAre(
                 NodeIs(0, {{0, 10}, {1, 10}, {2, 10}, {3, 1}, {4, 1}, {5, 1}}),
                 NodeIs(1, {{1, 10}, {2, 10}, {5, 1}}), NodeIs(2, {{2, 10}}),
                 NodeIs(3, {{2, 10}}), NodeIs(4, {{4, 1}, {5, 1}})));
}

} // end namespace llvm
