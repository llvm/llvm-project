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
using testing::Matcher;
using testing::UnorderedElementsAre;

namespace llvm {

void PrintTo(const BPFunctionNode &Node, std::ostream *OS) {
  raw_os_ostream ROS(*OS);
  Node.dump(ROS);
}

TEST(BPFunctionNodeTest, Basic) {
  auto UNIdsAre = [](auto... Ids) {
    return UnorderedElementsAre(Field("Id", &BPFunctionNode::UtilityNodeT::Id,
                                      std::forward<uint32_t>(Ids))...);
  };
  auto NodeIs = [](BPFunctionNode::IDT Id,
                   Matcher<ArrayRef<BPFunctionNode::UtilityNodeT>> UNsMatcher) {
    return AllOf(
        Field("Id", &BPFunctionNode::Id, Id),
        Field("UtilityNodes", &BPFunctionNode::UtilityNodes, UNsMatcher));
  };

  auto Nodes = TemporalProfTraceTy::createBPFunctionNodes({
      TemporalProfTraceTy({0, 1, 2, 3}),
  });
  EXPECT_THAT(Nodes, UnorderedElementsAre(NodeIs(0, UNIdsAre(0, 1, 2)),
                                          NodeIs(1, UNIdsAre(1, 2)),
                                          NodeIs(2, UNIdsAre(1, 2)),
                                          NodeIs(3, UNIdsAre(2))));

  Nodes = TemporalProfTraceTy::createBPFunctionNodes({
      TemporalProfTraceTy({0, 1, 2, 3, 4}),
      TemporalProfTraceTy({4, 2}),
  });

  EXPECT_THAT(Nodes, UnorderedElementsAre(NodeIs(0, UNIdsAre(0, 1, 2)),
                                          NodeIs(1, UNIdsAre(1, 2)),
                                          NodeIs(2, UNIdsAre(1, 2, 4, 5)),
                                          NodeIs(3, UNIdsAre(2)),
                                          NodeIs(4, UNIdsAre(2, 3, 4, 5))));
}

} // end namespace llvm
