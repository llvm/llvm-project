//===- llvm/unittest/ADT/DeltaAlgorithmTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DeltaAlgorithm.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdarg>
using namespace llvm;

namespace std {

std::ostream &operator<<(std::ostream &OS,
                         const std::set<unsigned> &S) {
  OS << "{";
  for (std::set<unsigned>::const_iterator it = S.begin(),
         ie = S.end(); it != ie; ++it) {
    if (it != S.begin())
      OS << ",";
    OS << *it;
  }
  OS << "}";
  return OS;
}

}

namespace {

class FixedDeltaAlgorithm final : public DeltaAlgorithm {
  changeset_ty FailingSet;
  unsigned NumTests;

protected:
  bool ExecuteOneTest(const changeset_ty &Changes) override {
    ++NumTests;
    return std::includes(Changes.begin(), Changes.end(),
                         FailingSet.begin(), FailingSet.end());
  }

public:
  FixedDeltaAlgorithm(const changeset_ty &_FailingSet)
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

TEST(DeltaAlgorithmTest, Basic) {
  // P = {3,5,7} \in S
  //   [0, 20) should minimize to {3,5,7} in a reasonable number of tests.
  std::set<unsigned> Fails = fixed_set(3, 3, 5, 7);
  FixedDeltaAlgorithm FDA(Fails);
  EXPECT_EQ(fixed_set(3, 3, 5, 7), FDA.Run(range(20)));
  EXPECT_GE(33U, FDA.getNumTests());

  // P = {3,5,7} \in S
  //   [10, 20) should minimize to [10,20)
  EXPECT_EQ(range(10,20), FDA.Run(range(10,20)));

  // P = [0,4) \in S
  //   [0, 4) should minimize to [0,4) in 11 tests.
  //
  // 11 = |{ {},
  //         {0}, {1}, {2}, {3},
  //         {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}, 
  //         {0, 1}, {2, 3} }|
  FDA = FixedDeltaAlgorithm(range(10));
  EXPECT_EQ(range(4), FDA.Run(range(4)));
  EXPECT_EQ(11U, FDA.getNumTests());  
}

}

