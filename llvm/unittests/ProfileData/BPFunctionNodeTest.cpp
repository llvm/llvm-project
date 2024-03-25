//===- BPFunctionNodeTest.cpp - BPFunctionNode tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/BalancedPartitioning.h"
#include "llvm/Testing/Support/SupportHelpers.h"
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

  auto Nodes = TemporalProfTraceTy::createBPFunctionNodes({
      TemporalProfTraceTy({0, 1, 2, 3}),
  });
  EXPECT_THAT(Nodes,
              UnorderedElementsAre(NodeIs(0, {0, 1, 2}), NodeIs(1, {1, 2}),
                                   NodeIs(2, {1, 2}), NodeIs(3, {2})));

  Nodes = TemporalProfTraceTy::createBPFunctionNodes({
      TemporalProfTraceTy({0, 1, 2, 3, 4}),
      TemporalProfTraceTy({4, 2}),
  });

  EXPECT_THAT(Nodes,
              UnorderedElementsAre(NodeIs(0, {0, 1, 2}), NodeIs(1, {1, 2}),
                                   NodeIs(2, {1, 2, 4, 5}), NodeIs(3, {2}),
                                   NodeIs(4, {2, 3, 4, 5})));
}

} // end namespace llvm
