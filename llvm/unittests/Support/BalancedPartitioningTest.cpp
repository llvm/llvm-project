//===- BalancedPartitioningTest.cpp - BalancedPartitioning tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BalancedPartitioning.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::Each;
using testing::Field;
using testing::Not;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

namespace llvm {

void PrintTo(const BPFunctionNode &Node, std::ostream *OS) {
  raw_os_ostream ROS(*OS);
  Node.dump(ROS);
}

class BalancedPartitioningTest : public ::testing::Test {
protected:
  BalancedPartitioningConfig Config;
  BalancedPartitioning Bp;
  BalancedPartitioningTest() : Bp(Config) {}

  static std::vector<BPFunctionNode::IDT>
  getIds(std::vector<BPFunctionNode> Nodes) {
    std::vector<BPFunctionNode::IDT> Ids;
    for (auto &N : Nodes)
      Ids.push_back(N.Id);
    return Ids;
  }
};

TEST_F(BalancedPartitioningTest, Basic) {
  std::vector<BPFunctionNode> Nodes = {
      BPFunctionNode(0, {1, 2}), BPFunctionNode(2, {3, 4}),
      BPFunctionNode(1, {1, 2}), BPFunctionNode(3, {3, 4}),
      BPFunctionNode(4, {4}),
  };

  Bp.run(Nodes);

  auto NodeIs = [](BPFunctionNode::IDT Id, std::optional<uint32_t> Bucket) {
    return AllOf(Field("Id", &BPFunctionNode::Id, Id),
                 Field("Bucket", &BPFunctionNode::Bucket, Bucket));
  };

  EXPECT_THAT(Nodes,
              UnorderedElementsAre(NodeIs(0, 0), NodeIs(1, 1), NodeIs(2, 2),
                                   NodeIs(3, 3), NodeIs(4, 4)));
}

TEST_F(BalancedPartitioningTest, Large) {
  const int ProblemSize = 1000;
  std::vector<BPFunctionNode::UtilityNodeT> AllUNs;
  for (int i = 0; i < ProblemSize; i++)
    AllUNs.emplace_back(i);

  std::mt19937 RNG;
  std::vector<BPFunctionNode> Nodes;
  for (int i = 0; i < ProblemSize; i++) {
    std::vector<BPFunctionNode::UtilityNodeT> UNs;
    int SampleSize =
        std::uniform_int_distribution<int>(0, AllUNs.size() - 1)(RNG);
    std::sample(AllUNs.begin(), AllUNs.end(), std::back_inserter(UNs),
                SampleSize, RNG);
    Nodes.emplace_back(i, UNs);
  }

  auto OrigIds = getIds(Nodes);

  Bp.run(Nodes);

  EXPECT_THAT(
      Nodes, Each(Not(Field("Bucket", &BPFunctionNode::Bucket, std::nullopt))));
  EXPECT_THAT(getIds(Nodes), UnorderedElementsAreArray(OrigIds));
}

TEST_F(BalancedPartitioningTest, MoveGain) {
  BalancedPartitioning::SignaturesT Signatures = {
      {10, 10, 10.f, 0.f, true}, // 0
      {10, 10, 0.f, 10.f, true}, // 1
      {10, 10, 0.f, 20.f, true}, // 2
  };
  EXPECT_FLOAT_EQ(Bp.moveGain(BPFunctionNode(0, {}), true, Signatures), 0.f);
  EXPECT_FLOAT_EQ(Bp.moveGain(BPFunctionNode(0, {0, 1}), true, Signatures),
                  10.f);
  EXPECT_FLOAT_EQ(Bp.moveGain(BPFunctionNode(0, {1, 2}), false, Signatures),
                  30.f);
}

} // end namespace llvm
