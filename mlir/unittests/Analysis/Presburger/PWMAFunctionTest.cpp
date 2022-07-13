//===- PWMAFunctionTest.cpp - Tests for PWMAFunction ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for PWMAFunction.
//
//===----------------------------------------------------------------------===//

#include "./Utils.h"

#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/IR/MLIRContext.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

using testing::ElementsAre;

TEST(PWAFunctionTest, isEqual) {
  // The output expressions are different but it doesn't matter because they are
  // equal in this domain.
  PWMAFunction idAtZeros = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (y == 0)", {{1, 0, 0}, {0, 1, 0}}},             // (x, y).
          {"(x, y) : (y - 1 >= 0, x == 0)", {{1, 0, 0}, {0, 1, 0}}}, // (x, y).
          {"(x, y) : (-y - 1 >= 0, x == 0)", {{1, 0, 0}, {0, 1, 0}}} // (x, y).
      });
  PWMAFunction idAtZeros2 = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (y == 0)", {{1, 0, 0}, {0, 20, 0}}}, // (x, 20y).
          {"(x, y) : (y - 1 >= 0, x == 0)", {{30, 0, 0}, {0, 1, 0}}}, //(30x, y)
          {"(x, y) : (-y - 1 > =0, x == 0)", {{30, 0, 0}, {0, 1, 0}}} //(30x, y)
      });
  EXPECT_TRUE(idAtZeros.isEqual(idAtZeros2));

  PWMAFunction notIdAtZeros = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (y == 0)", {{1, 0, 0}, {0, 1, 0}}},              // (x, y).
          {"(x, y) : (y - 1 >= 0, x == 0)", {{1, 0, 0}, {0, 2, 0}}},  // (x, 2y)
          {"(x, y) : (-y - 1 >= 0, x == 0)", {{1, 0, 0}, {0, 2, 0}}}, // (x, 2y)
      });
  EXPECT_FALSE(idAtZeros.isEqual(notIdAtZeros));

  // These match at their intersection but one has a bigger domain.
  PWMAFunction idNoNegNegQuadrant = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (x >= 0)", {{1, 0, 0}, {0, 1, 0}}},             // (x, y).
          {"(x, y) : (-x - 1 >= 0, y >= 0)", {{1, 0, 0}, {0, 1, 0}}} // (x, y).
      });
  PWMAFunction idOnlyPosX =
      parsePWMAF(/*numInputs=*/2, /*numOutputs=*/2,
                 {
                     {"(x, y) : (x >= 0)", {{1, 0, 0}, {0, 1, 0}}}, // (x, y).
                 });
  EXPECT_FALSE(idNoNegNegQuadrant.isEqual(idOnlyPosX));

  // Different representations of the same domain.
  PWMAFunction sumPlusOne = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x >= 0)", {{1, 1, 1}}},                   // x + y + 1.
          {"(x, y) : (-x - 1 >= 0, -y - 1 >= 0)", {{1, 1, 1}}}, // x + y + 1.
          {"(x, y) : (-x - 1 >= 0, y >= 0)", {{1, 1, 1}}}       // x + y + 1.
      });
  PWMAFunction sumPlusOne2 =
      parsePWMAF(/*numInputs=*/2, /*numOutputs=*/1,
                 {
                     {"(x, y) : ()", {{1, 1, 1}}}, // x + y + 1.
                 });
  EXPECT_TRUE(sumPlusOne.isEqual(sumPlusOne2));

  // Functions with zero input dimensions.
  PWMAFunction noInputs1 = parsePWMAF(/*numInputs=*/0, /*numOutputs=*/1,
                                      {
                                          {"() : ()", {{1}}}, // 1.
                                      });
  PWMAFunction noInputs2 = parsePWMAF(/*numInputs=*/0, /*numOutputs=*/1,
                                      {
                                          {"() : ()", {{2}}}, // 1.
                                      });
  EXPECT_TRUE(noInputs1.isEqual(noInputs1));
  EXPECT_FALSE(noInputs1.isEqual(noInputs2));

  // Mismatched dimensionalities.
  EXPECT_FALSE(noInputs1.isEqual(sumPlusOne));
  EXPECT_FALSE(idOnlyPosX.isEqual(sumPlusOne));

  // Divisions.
  // Domain is only multiples of 6; x = 6k for some k.
  // x + 4(x/2) + 4(x/3) == 26k.
  PWMAFunction mul2AndMul3 = parsePWMAF(
      /*numInputs=*/1, /*numOutputs=*/1,
      {
          {"(x) : (x - 2*(x floordiv 2) == 0, x - 3*(x floordiv 3) == 0)",
           {{1, 4, 4, 0}}}, // x + 4(x/2) + 4(x/3).
      });
  PWMAFunction mul6 = parsePWMAF(
      /*numInputs=*/1, /*numOutputs=*/1,
      {
          {"(x) : (x - 6*(x floordiv 6) == 0)", {{0, 26, 0}}}, // 26(x/6).
      });
  EXPECT_TRUE(mul2AndMul3.isEqual(mul6));

  PWMAFunction mul6diff = parsePWMAF(
      /*numInputs=*/1, /*numOutputs=*/1,
      {
          {"(x) : (x - 5*(x floordiv 5) == 0)", {{0, 52, 0}}}, // 52(x/6).
      });
  EXPECT_FALSE(mul2AndMul3.isEqual(mul6diff));

  PWMAFunction mul5 = parsePWMAF(
      /*numInputs=*/1, /*numOutputs=*/1,
      {
          {"(x) : (x - 5*(x floordiv 5) == 0)", {{0, 26, 0}}}, // 26(x/5).
      });
  EXPECT_FALSE(mul2AndMul3.isEqual(mul5));
}

TEST(PWMAFunction, valueAt) {
  PWMAFunction nonNegPWMAF = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (x >= 0)", {{1, 2, 3}, {3, 4, 5}}}, // (x, y).
          {"(x, y) : (y >= 0, -x - 1 >= 0)", {{-1, 2, 3}, {-3, 4, 5}}} // (x, y)
      });
  EXPECT_THAT(*nonNegPWMAF.valueAt({2, 3}), ElementsAre(11, 23));
  EXPECT_THAT(*nonNegPWMAF.valueAt({-2, 3}), ElementsAre(11, 23));
  EXPECT_THAT(*nonNegPWMAF.valueAt({2, -3}), ElementsAre(-1, -1));
  EXPECT_FALSE(nonNegPWMAF.valueAt({-2, -3}).has_value());

  PWMAFunction divPWMAF = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (x >= 0, x - 2*(x floordiv 2) == 0)",
           {{0, 2, 1, 3}, {0, 4, 3, 5}}}, // (x, y).
          {"(x, y) : (y >= 0, -x - 1 >= 0)", {{-1, 2, 3}, {-3, 4, 5}}} // (x, y)
      });
  EXPECT_THAT(*divPWMAF.valueAt({4, 3}), ElementsAre(11, 23));
  EXPECT_THAT(*divPWMAF.valueAt({4, -3}), ElementsAre(-1, -1));
  EXPECT_FALSE(divPWMAF.valueAt({3, 3}).has_value());
  EXPECT_FALSE(divPWMAF.valueAt({3, -3}).has_value());

  EXPECT_THAT(*divPWMAF.valueAt({-2, 3}), ElementsAre(11, 23));
  EXPECT_FALSE(divPWMAF.valueAt({-2, -3}).has_value());
}

TEST(PWMAFunction, removeIdRangeRegressionTest) {
  PWMAFunction pwmafA = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x == 0, y == 0, x - 2*(x floordiv 2) == 0, y - 2*(y "
           "floordiv 2) == 0)",
           {{0, 0, 0, 0, 0}}} // (0, 0)
      });
  PWMAFunction pwmafB = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x - 11*y == 0, 11*x - y == 0, x - 2*(x floordiv 2) == 0, "
           "y - 2*(y floordiv 2) == 0)",
           {{0, 0, 0, 0, 0}}} // (0, 0)
      });
  EXPECT_TRUE(pwmafA.isEqual(pwmafB));
}

TEST(PWMAFunction, eliminateRedundantLocalIdRegressionTest) {
  PWMAFunction pwmafA = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x - 2*(x floordiv 2) == 0, x - 2*y == 0)",
           {{0, 1, 0, 0}}} // (0, 0)
      });
  PWMAFunction pwmafB = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x - 2*(x floordiv 2) == 0, x - 2*y == 0)",
           {{1, -1, 0, 0}}} // (0, 0)
      });
  EXPECT_TRUE(pwmafA.isEqual(pwmafB));
}

TEST(PWMAFunction, unionLexMaxSimple) {
  // func2 is better than func1, but func2's domain is empty.
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : ()", {{0, 1}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (1 == 0)", {{0, 2}}},
        });

    EXPECT_TRUE(func1.unionLexMax(func2).isEqual(func1));
    EXPECT_TRUE(func2.unionLexMax(func1).isEqual(func1));
  }

  // func2 is better than func1 on a subset of func1.
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : ()", {{0, 1}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (x >= 0, 10 - x >= 0)", {{0, 2}}},
        });

    PWMAFunction result = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (-1 - x >= 0)", {{0, 1}}},
            {"(x) : (x >= 0, 10 - x >= 0)", {{0, 2}}},
            {"(x) : (x - 11 >= 0)", {{0, 1}}},
        });

    EXPECT_TRUE(func1.unionLexMax(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMax(func1).isEqual(result));
  }

  // func1 and func2 are defined over the whole domain with different outputs.
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : ()", {{1, 0}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : ()", {{-1, 0}}},
        });

    PWMAFunction result = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (x >= 0)", {{1, 0}}},
            {"(x) : (-1 - x >= 0)", {{-1, 0}}},
        });

    EXPECT_TRUE(func1.unionLexMax(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMax(func1).isEqual(result));
  }

  // func1 and func2 have disjoint domains.
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (x >= 0, 10 - x >= 0)", {{0, 1}}},
            {"(x) : (x - 71 >= 0, 80 - x >= 0)", {{0, 1}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (x - 20 >= 0, 41 - x >= 0)", {{0, 2}}},
            {"(x) : (x - 101 >= 0, 120 - x >= 0)", {{0, 2}}},
        });

    PWMAFunction result = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (x >= 0, 10 - x >= 0)", {{0, 1}}},
            {"(x) : (x - 71 >= 0, 80 - x >= 0)", {{0, 1}}},
            {"(x) : (x - 20 >= 0, 41 - x >= 0)", {{0, 2}}},
            {"(x) : (x - 101 >= 0, 120 - x >= 0)", {{0, 2}}},
        });

    EXPECT_TRUE(func1.unionLexMin(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMin(func1).isEqual(result));
  }
}

TEST(PWMAFunction, unionLexMinSimple) {
  // func2 is better than func1, but func2's domain is empty.
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : ()", {{0, -1}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (1 == 0)", {{0, -2}}},
        });

    EXPECT_TRUE(func1.unionLexMin(func2).isEqual(func1));
    EXPECT_TRUE(func2.unionLexMin(func1).isEqual(func1));
  }

  // func2 is better than func1 on a subset of func1.
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : ()", {{0, -1}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (x >= 0, 10 - x >= 0)", {{0, -2}}},
        });

    PWMAFunction result = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (-1 - x >= 0)", {{0, -1}}},
            {"(x) : (x >= 0, 10 - x >= 0)", {{0, -2}}},
            {"(x) : (x - 11 >= 0)", {{0, -1}}},
        });

    EXPECT_TRUE(func1.unionLexMin(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMin(func1).isEqual(result));
  }

  // func1 and func2 are defined over the whole domain with different outputs.
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : ()", {{-1, 0}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : ()", {{1, 0}}},
        });

    PWMAFunction result = parsePWMAF(
        /*numInputs=*/1, /*numOutputs=*/1,
        {
            {"(x) : (x >= 0)", {{-1, 0}}},
            {"(x) : (-1 - x >= 0)", {{1, 0}}},
        });

    EXPECT_TRUE(func1.unionLexMin(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMin(func1).isEqual(result));
  }
}

TEST(PWMAFunction, unionLexMaxComplex) {
  // Union of function containing 4 different pieces of output.
  //
  // x >= 21               --> func1 (func2 not defined)
  // x <= 0                --> func2 (func1 not defined)
  // 10 <= x <= 20, y >  0 --> func1 (x + y  > x - y for y >  0)
  // 10 <= x <= 20, y <= 0 --> func2 (x + y <= x - y for y <= 0)
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/2, /*numOutputs=*/1,
        {
            {"(x, y) : (x >= 10)", {{1, 1, 0}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/2, /*numOutputs=*/1,
        {
            {"(x, y) : (x <= 20)", {{1, -1, 0}}},
        });

    PWMAFunction result = parsePWMAF(/*numInputs=*/2, /*numOutputs=*/1,
                                     {{"(x, y) : (x >= 10, x <= 20, y >= 1)",
                                       {
                                           {1, 1, 0},
                                       }},
                                      {"(x, y) : (x >= 21)",
                                       {
                                           {1, 1, 0},
                                       }},
                                      {"(x, y) : (x <= 9)",
                                       {
                                           {1, -1, 0},
                                       }},
                                      {"(x, y) : (x >= 10, x <= 20, y <= 0)",
                                       {
                                           {1, -1, 0},
                                       }}});

    EXPECT_TRUE(func1.unionLexMax(func2).isEqual(result));
  }

  // Functions with more than one output, with contribution from both functions.
  //
  // If y >= 1, func1 is better because in the first output,
  // x + y (func1) > x (func2), when y >= 1
  //
  // If y == 0, the first output is same for both functions, so we look at the
  // second output. -2x + 4 (func1) > 2x - 2 (func2) when 0 <= x <= 1, so we
  // take func1 for this domain and func2 for the remaining.
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/2, /*numOutputs=*/2,
        {
            {"(x, y) : (x >= 0, y >= 0)", {{1, 1, 0}, {-2, 0, 4}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/2, /*numOutputs=*/2,
        {
            {"(x, y) : (x >= 0, y >= 0)", {{1, 0, 0}, {2, 0, -2}}},
        });

    PWMAFunction result = parsePWMAF(/*numInputs=*/2, /*numOutputs=*/2,
                                     {{"(x, y) : (x >= 0, y >= 1)",
                                       {
                                           {1, 1, 0},
                                           {-2, 0, 4},
                                       }},
                                      {"(x, y) : (x >= 0, x <= 1, y == 0)",
                                       {
                                           {1, 1, 0},
                                           {-2, 0, 4},
                                       }},
                                      {"(x, y) : (x >= 2, y == 0)",
                                       {
                                           {1, 0, 0},
                                           {2, 0, -2},
                                       }}});

    EXPECT_TRUE(func1.unionLexMax(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMax(func1).isEqual(result));
  }

  // Function with three boolean variables `a, b, c` used to control which
  // output will be taken lexicographically.
  //
  // a == 1                 --> Take func2
  // a == 0, b == 1         --> Take func1
  // a == 0, b == 0, c == 1 --> Take func2
  {
    PWMAFunction func1 = parsePWMAF(
        /*numInputs=*/3, /*numOutputs=*/3,
        {
            {"(a, b, c) : (a >= 0, 1 - a >= 0, b >= 0, 1 - b >= 0, c "
             ">= 0, 1 - c >= 0)",
             {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}}},
        });

    PWMAFunction func2 = parsePWMAF(
        /*numInputs=*/3, /*numOutputs=*/3,
        {
            {"(a, b, c) : (a >= 0, 1 - a >= 0, b >= 0, 1 - b >= 0, c >= 0, 1 - "
             "c >= 0)",
             {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}}},
        });

    PWMAFunction result = parsePWMAF(
        /*numInputs=*/3, /*numOutputs=*/3,
        {
            {"(a, b, c) : (a - 1 == 0, b >= 0, 1 - b >= 0, c >= 0, 1 - c >= 0)",
             {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}}},
            {"(a, b, c) : (a == 0, b - 1 == 0, c >= 0, 1 - c >= 0)",
             {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}}},
            {"(a, b, c) : (a == 0, b == 0, c >= 0, 1 - c >= 0)",
             {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}}},
        });

    EXPECT_TRUE(func1.unionLexMax(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMax(func1).isEqual(result));
  }
}

TEST(PWMAFunction, unionLexMinComplex) {
  // Regression test checking if lexicographic tiebreak produces disjoint
  // domains.
  //
  // If x == 1, func1 is better since in the first output,
  // -x (func1) is < 0 (func2) when x == 1.
  //
  // If x == 0, func1 and func2 both have the same first output. So we take a
  // look at the second output. func2 is better since in the second output,
  // y - 1 (func2) is < y (func1).
  PWMAFunction func1 = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (x >= 0, x <= 1, y >= 0, y <= 1)",
           {{-1, 0, 0}, {0, 1, 0}}},
      });

  PWMAFunction func2 = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (x >= 0, x <= 1, y >= 0, y <= 1)",
           {{0, 0, 0}, {0, 1, -1}}},
      });

  PWMAFunction result = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (x == 1, y >= 0, y <= 1)", {{-1, 0, 0}, {0, 1, 0}}},
          {"(x, y) : (x == 0, y >= 0, y <= 1)", {{0, 0, 0}, {0, 1, -1}}},
      });

  EXPECT_TRUE(func1.unionLexMin(func2).isEqual(result));
  EXPECT_TRUE(func2.unionLexMin(func1).isEqual(result));
}
