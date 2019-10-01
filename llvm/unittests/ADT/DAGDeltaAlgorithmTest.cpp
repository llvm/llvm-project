//===- llvm/unittest/ADT/DAGDeltaAlgorithmTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DAGDeltaAlgorithm.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdarg>
using namespace llvm;

namespace {

typedef DAGDeltaAlgorithm::edge_ty edge_ty;

class FixedDAGDeltaAlgorithm : public DAGDeltaAlgorithm {
  changeset_ty FailingSet;
  unsigned NumTests;

protected:
  bool ExecuteOneTest(const changeset_ty &Changes) override {
    ++NumTests;
    return std::includes(Changes.begin(), Changes.end(),
                         FailingSet.begin(), FailingSet.end());
  }

public:
  FixedDAGDeltaAlgorithm(const changeset_ty &_FailingSet)
    : FailingSet(_FailingSet),
      NumTests(0) {}

  unsigned getNumTests() const { return NumTests; }
};

std::set<unsigned> fixed_set(unsigned N, ...) {
  std::set<unsigned> S;
  va_list ap;
  va_start(ap, N);
  for (unsigned i = 0; i != N; ++i)
    S.insert(va_arg(ap, unsigned));
  va_end(ap);
  return S;
}

std::set<unsigned> range(unsigned Start, unsigned End) {
  std::set<unsigned> S;
  while (Start != End)
    S.insert(Start++);
  return S;
}

std::set<unsigned> range(unsigned N) {
  return range(0, N);
}

TEST(DAGDeltaAlgorithmTest, Basic) {
  std::vector<edge_ty> Deps;

  // Dependencies:
  //  1 - 3
  Deps.clear();
  Deps.push_back(std::make_pair(3, 1));

  // P = {3,5,7} \in S,
  //   [0, 20),
  // should minimize to {1,3,5,7} in a reasonable number of tests.
  FixedDAGDeltaAlgorithm FDA(fixed_set(3, 3, 5, 7));
  EXPECT_EQ(fixed_set(4, 1, 3, 5, 7), FDA.Run(range(20), Deps));
  EXPECT_GE(46U, FDA.getNumTests());

  // Dependencies:
  // 0 - 1
  //  \- 2 - 3
  //  \- 4
  Deps.clear();
  Deps.push_back(std::make_pair(1, 0));
  Deps.push_back(std::make_pair(2, 0));
  Deps.push_back(std::make_pair(4, 0));
  Deps.push_back(std::make_pair(3, 2));

  // This is a case where we must hold required changes.
  //
  // P = {1,3} \in S,
  //   [0, 5),
  // should minimize to {0,1,2,3} in a small number of tests.
  FixedDAGDeltaAlgorithm FDA2(fixed_set(2, 1, 3));
  EXPECT_EQ(fixed_set(4, 0, 1, 2, 3), FDA2.Run(range(5), Deps));
  EXPECT_GE(9U, FDA2.getNumTests());

  // This is a case where we should quickly prune part of the tree.
  //
  // P = {4} \in S,
  //   [0, 5),
  // should minimize to {0,4} in a small number of tests.
  FixedDAGDeltaAlgorithm FDA3(fixed_set(1, 4));
  EXPECT_EQ(fixed_set(2, 0, 4), FDA3.Run(range(5), Deps));
  EXPECT_GE(6U, FDA3.getNumTests());
}

}

