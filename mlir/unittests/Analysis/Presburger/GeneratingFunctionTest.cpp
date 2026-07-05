//===- MatrixTest.cpp - Tests for QuasiPolynomial -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/GeneratingFunction.h"
#include "./Utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;
using namespace mlir::presburger::detail;

TEST(GeneratingFunctionTest, sum) {
  GeneratingFunction gf1(2, {1, -1},
                         {makeFracMatrix(3, 2, {{1, 2}, {5, 7}, {2, 6}}),
                          makeFracMatrix(3, 2, {{5, 2}, {5, 3}, {7, 2}})},
                         {{{3, 6}, {7, 2}}, {{2, 8}, {6, 3}}});
  GeneratingFunction gf2(2, {1, 1},
                         {makeFracMatrix(3, 2, {{6, 2}, {1, 4}, {2, 6}}),
                          makeFracMatrix(3, 2, {{3, 2}, {6, 9}, {2, 5}})},
                         {{{3, 7}, {5, 1}}, {{5, 2}, {6, 2}}});

  GeneratingFunction sum = gf1 + gf2;
  EXPECT_EQ_REPR_GENERATINGFUNCTION(
      sum, GeneratingFunction(2, {1, -1, 1, 1},
                              {makeFracMatrix(3, 2, {{1, 2}, {5, 7}, {2, 6}}),
                               makeFracMatrix(3, 2, {{5, 2}, {5, 3}, {7, 2}}),
                               makeFracMatrix(3, 2, {{6, 2}, {1, 4}, {2, 6}}),
                               makeFracMatrix(3, 2, {{3, 2}, {6, 9}, {2, 5}})},
                              {{{3, 6}, {7, 2}},
                               {{2, 8}, {6, 3}},
                               {{3, 7}, {5, 1}},
                               {{5, 2}, {6, 2}}}));
}
