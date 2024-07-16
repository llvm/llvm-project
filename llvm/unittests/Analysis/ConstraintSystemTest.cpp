//===--- ConstraintSystemTests.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ConstraintSystem.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"
#include <initializer_list>

using namespace llvm;

namespace {
SmallVector<DynamicAPInt> toDynamicAPIntVec(std::initializer_list<int64_t> IL) {
  SmallVector<DynamicAPInt> Ret;
  Ret.reserve(IL.size());
  for (auto El : IL)
    Ret.emplace_back(El);
  return Ret;
}

TEST(ConstraintSolverTest, TestSolutionChecks) {
  {
    ConstraintSystem CS;
    // x + y <= 10, x >= 5, y >= 6, x <= 10, y <= 10
    CS.addVariableRow(toDynamicAPIntVec({10, 1, 1}));
    CS.addVariableRow(toDynamicAPIntVec({-5, -1, 0}));
    CS.addVariableRow(toDynamicAPIntVec({-6, 0, -1}));
    CS.addVariableRow(toDynamicAPIntVec({10, 1, 0}));
    CS.addVariableRow(toDynamicAPIntVec({10, 0, 1}));

    EXPECT_FALSE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;
    // x + y <= 10, x >= 2, y >= 3, x <= 10, y <= 10
    CS.addVariableRow(toDynamicAPIntVec({10, 1, 1}));
    CS.addVariableRow(toDynamicAPIntVec({-2, -1, 0}));
    CS.addVariableRow(toDynamicAPIntVec({-3, 0, -1}));
    CS.addVariableRow(toDynamicAPIntVec({10, 1, 0}));
    CS.addVariableRow(toDynamicAPIntVec({10, 0, 1}));

    EXPECT_TRUE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;
    // x + y <= 10, x >= 10, y >= 10; does not have a solution.
    CS.addVariableRow(toDynamicAPIntVec({10, 1, 1}));
    CS.addVariableRow(toDynamicAPIntVec({-10, -1, 0}));
    CS.addVariableRow(toDynamicAPIntVec({-10, 0, -1}));

    EXPECT_FALSE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;
    // x + y >= 20, 10 >= x, 10 >= y; does HAVE a solution.
    CS.addVariableRow(toDynamicAPIntVec({-20, -1, -1}));
    CS.addVariableRow(toDynamicAPIntVec({-10, -1, 0}));
    CS.addVariableRow(toDynamicAPIntVec({-10, 0, -1}));

    EXPECT_TRUE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;

    // 2x + y + 3z <= 10,  2x + y >= 10, y >= 1
    CS.addVariableRow(toDynamicAPIntVec({10, 2, 1, 3}));
    CS.addVariableRow(toDynamicAPIntVec({-10, -2, -1, 0}));
    CS.addVariableRow(toDynamicAPIntVec({-1, 0, 0, -1}));

    EXPECT_FALSE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;

    // 2x + y + 3z <= 10,  2x + y >= 10
    CS.addVariableRow(toDynamicAPIntVec({10, 2, 1, 3}));
    CS.addVariableRow(toDynamicAPIntVec({-10, -2, -1, 0}));

    EXPECT_TRUE(CS.mayHaveSolution());
  }
}

TEST(ConstraintSolverTest, IsConditionImplied) {
  {
    // For the test below, we assume we know
    // x <= 5 && y <= 3
    ConstraintSystem CS;
    CS.addVariableRow(toDynamicAPIntVec({5, 1, 0}));
    CS.addVariableRow(toDynamicAPIntVec({3, 0, 1}));

    // x + y <= 6 does not hold.
    EXPECT_FALSE(CS.isConditionImplied(toDynamicAPIntVec({6, 1, 1})));
    // x + y <= 7 does not hold.
    EXPECT_FALSE(CS.isConditionImplied(toDynamicAPIntVec({7, 1, 1})));
    // x + y <= 8 does hold.
    EXPECT_TRUE(CS.isConditionImplied(toDynamicAPIntVec({8, 1, 1})));

    // 2 * x + y <= 12 does hold.
    EXPECT_FALSE(CS.isConditionImplied(toDynamicAPIntVec({12, 2, 1})));
    // 2 * x + y <= 13 does hold.
    EXPECT_TRUE(CS.isConditionImplied(toDynamicAPIntVec({13, 2, 1})));

    //  x + y <= 12 does hold.
    EXPECT_FALSE(CS.isConditionImplied(toDynamicAPIntVec({12, 2, 1})));
    // 2 * x + y <= 13 does hold.
    EXPECT_TRUE(CS.isConditionImplied(toDynamicAPIntVec({13, 2, 1})));

    // x <= y == x - y <= 0 does not hold.
    EXPECT_FALSE(CS.isConditionImplied(toDynamicAPIntVec({0, 1, -1})));
    // y <= x == -x + y <= 0 does not hold.
    EXPECT_FALSE(CS.isConditionImplied(toDynamicAPIntVec({0, -1, 1})));
  }

  {
    // For the test below, we assume we know
    // x + 1 <= y + 1 == x - y <= 0
    ConstraintSystem CS;
    CS.addVariableRow(toDynamicAPIntVec({0, 1, -1}));

    // x <= y == x - y <= 0 does hold.
    EXPECT_TRUE(CS.isConditionImplied(toDynamicAPIntVec({0, 1, -1})));
    // y <= x == -x + y <= 0 does not hold.
    EXPECT_FALSE(CS.isConditionImplied(toDynamicAPIntVec({0, -1, 1})));

    // x <= y + 10 == x - y <= 10 does hold.
    EXPECT_TRUE(CS.isConditionImplied(toDynamicAPIntVec({10, 1, -1})));
    // x + 10 <= y == x - y <= -10 does NOT hold.
    EXPECT_FALSE(CS.isConditionImplied(toDynamicAPIntVec({-10, 1, -1})));
  }

  {
    // For the test below, we assume we know
    // x <= y == x - y <= 0
    // y <= z == y - x <= 0
    ConstraintSystem CS;
    CS.addVariableRow(toDynamicAPIntVec({0, 1, -1, 0}));
    CS.addVariableRow(toDynamicAPIntVec({0, 0, 1, -1}));

    // z <= y == -y + z <= 0 does not hold.
    EXPECT_FALSE(CS.isConditionImplied(toDynamicAPIntVec({0, 0, -1, 1})));
    // x <= z == x - z <= 0 does hold.
    EXPECT_TRUE(CS.isConditionImplied(toDynamicAPIntVec({0, 1, 0, -1})));
  }
}

TEST(ConstraintSolverTest, IsConditionImpliedOverflow) {
  ConstraintSystem CS;
  // Make sure overflows are automatically handled by DynamicAPInt.
  int64_t Limit = std::numeric_limits<int64_t>::max();
  CS.addVariableRow(toDynamicAPIntVec({Limit - 1, Limit - 2, Limit - 3}));
  EXPECT_TRUE(CS.isConditionImplied(
      toDynamicAPIntVec({Limit - 1, Limit - 2, Limit - 3})));
}
} // namespace
