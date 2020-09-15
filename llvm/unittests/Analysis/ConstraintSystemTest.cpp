//===--- ConstraintSystemTests.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ConstraintSystem.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ConstraintSloverTest, TestSolutionChecks) {
  {
    ConstraintSystem CS;
    // x + y <= 10, x >= 5, y >= 6, x <= 10, y <= 10
    CS.addVariableRow({10, 1, 1});
    CS.addVariableRow({-5, -1, 0});
    CS.addVariableRow({-6, 0, -1});
    CS.addVariableRow({10, 1, 0});
    CS.addVariableRow({10, 0, 1});

    EXPECT_FALSE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;
    // x + y <= 10, x >= 2, y >= 3, x <= 10, y <= 10
    CS.addVariableRow({10, 1, 1});
    CS.addVariableRow({-2, -1, 0});
    CS.addVariableRow({-3, 0, -1});
    CS.addVariableRow({10, 1, 0});
    CS.addVariableRow({10, 0, 1});

    EXPECT_TRUE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;
    // x + y <= 10, 10 >= x, 10 >= y; does not have a solution.
    CS.addVariableRow({10, 1, 1});
    CS.addVariableRow({-10, -1, 0});
    CS.addVariableRow({-10, 0, -1});

    EXPECT_FALSE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;
    // x + y >= 20, 10 >= x, 10 >= y; does HAVE a solution.
    CS.addVariableRow({-20, -1, -1});
    CS.addVariableRow({-10, -1, 0});
    CS.addVariableRow({-10, 0, -1});

    EXPECT_TRUE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;

    // 2x + y + 3z <= 10,  2x + y >= 10, y >= 1
    CS.addVariableRow({10, 2, 1, 3});
    CS.addVariableRow({-10, -2, -1, 0});
    CS.addVariableRow({-1, 0, 0, -1});

    EXPECT_FALSE(CS.mayHaveSolution());
  }

  {
    ConstraintSystem CS;

    // 2x + y + 3z <= 10,  2x + y >= 10
    CS.addVariableRow({10, 2, 1, 3});
    CS.addVariableRow({-10, -2, -1, 0});

    EXPECT_TRUE(CS.mayHaveSolution());
  }
}
} // namespace
