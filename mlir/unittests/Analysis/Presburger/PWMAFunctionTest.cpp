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

#include "Parser.h"

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
  PWMAFunction idAtZeros =
      parsePWMAF({{"(x, y) : (y == 0)", "(x, y) -> (x, y)"},
                  {"(x, y) : (y - 1 >= 0, x == 0)", "(x, y) -> (x, y)"},
                  {"(x, y) : (-y - 1 >= 0, x == 0)", "(x, y) -> (x, y)"}});
  PWMAFunction idAtZeros2 =
      parsePWMAF({{"(x, y) : (y == 0)", "(x, y) -> (x, 20*y)"},
                  {"(x, y) : (y - 1 >= 0, x == 0)", "(x, y) -> (30*x, y)"},
                  {"(x, y) : (-y - 1 > =0, x == 0)", "(x, y) -> (30*x, y)"}});
  EXPECT_TRUE(idAtZeros.isEqual(idAtZeros2));

  PWMAFunction notIdAtZeros = parsePWMAF({
      {"(x, y) : (y == 0)", "(x, y) -> (x, y)"},
      {"(x, y) : (y - 1 >= 0, x == 0)", "(x, y) -> (x, 2*y)"},
      {"(x, y) : (-y - 1 >= 0, x == 0)", "(x, y) -> (x, 2*y)"},
  });
  EXPECT_FALSE(idAtZeros.isEqual(notIdAtZeros));

  // These match at their intersection but one has a bigger domain.
  PWMAFunction idNoNegNegQuadrant =
      parsePWMAF({{"(x, y) : (x >= 0)", "(x, y) -> (x, y)"},
                  {"(x, y) : (-x - 1 >= 0, y >= 0)", "(x, y) -> (x, y)"}});
  PWMAFunction idOnlyPosX = parsePWMAF({
      {"(x, y) : (x >= 0)", "(x, y) -> (x, y)"},
  });
  EXPECT_FALSE(idNoNegNegQuadrant.isEqual(idOnlyPosX));

  // Different representations of the same domain.
  PWMAFunction sumPlusOne = parsePWMAF({
      {"(x, y) : (x >= 0)", "(x, y) -> (x + y + 1)"},
      {"(x, y) : (-x - 1 >= 0, -y - 1 >= 0)", "(x, y) -> (x + y + 1)"},
      {"(x, y) : (-x - 1 >= 0, y >= 0)", "(x, y) -> (x + y + 1)"},
  });
  PWMAFunction sumPlusOne2 = parsePWMAF({
      {"(x, y) : ()", "(x, y) -> (x + y + 1)"},
  });
  EXPECT_TRUE(sumPlusOne.isEqual(sumPlusOne2));

  // Functions with zero input dimensions.
  PWMAFunction noInputs1 = parsePWMAF({
      {"() : ()", "() -> (1)"},
  });
  PWMAFunction noInputs2 = parsePWMAF({
      {"() : ()", "() -> (2)"},
  });
  EXPECT_TRUE(noInputs1.isEqual(noInputs1));
  EXPECT_FALSE(noInputs1.isEqual(noInputs2));

  // Mismatched dimensionalities.
  EXPECT_FALSE(noInputs1.isEqual(sumPlusOne));
  EXPECT_FALSE(idOnlyPosX.isEqual(sumPlusOne));

  // Divisions.
  // Domain is only multiples of 6; x = 6k for some k.
  // x + 4(x/2) + 4(x/3) == 26k.
  PWMAFunction mul2AndMul3 = parsePWMAF({
      {"(x) : (x - 2*(x floordiv 2) == 0, x - 3*(x floordiv 3) == 0)",
       "(x) -> (x + 4 * (x floordiv 2) + 4 * (x floordiv 3))"},
  });
  PWMAFunction mul6 = parsePWMAF({
      {"(x) : (x - 6*(x floordiv 6) == 0)", "(x) -> (26 * (x floordiv 6))"},
  });
  EXPECT_TRUE(mul2AndMul3.isEqual(mul6));

  PWMAFunction mul6diff = parsePWMAF({
      {"(x) : (x - 5*(x floordiv 5) == 0)", "(x) -> (52 * (x floordiv 6))"},
  });
  EXPECT_FALSE(mul2AndMul3.isEqual(mul6diff));

  PWMAFunction mul5 = parsePWMAF({
      {"(x) : (x - 5*(x floordiv 5) == 0)", "(x) -> (26 * (x floordiv 5))"},
  });
  EXPECT_FALSE(mul2AndMul3.isEqual(mul5));
}

TEST(PWMAFunction, valueAt) {
  PWMAFunction nonNegPWMAF = parsePWMAF(
      {{"(x, y) : (x >= 0)", "(x, y) -> (x + 2*y + 3, 3*x + 4*y + 5)"},
       {"(x, y) : (y >= 0, -x - 1 >= 0)",
        "(x, y) -> (-x + 2*y + 3, -3*x + 4*y + 5)"}});
  EXPECT_THAT(*nonNegPWMAF.valueAt({2, 3}), ElementsAre(11, 23));
  EXPECT_THAT(*nonNegPWMAF.valueAt({-2, 3}), ElementsAre(11, 23));
  EXPECT_THAT(*nonNegPWMAF.valueAt({2, -3}), ElementsAre(-1, -1));
  EXPECT_FALSE(nonNegPWMAF.valueAt({-2, -3}).has_value());

  PWMAFunction divPWMAF = parsePWMAF(
      {{"(x, y) : (x >= 0, x - 2*(x floordiv 2) == 0)",
        "(x, y) -> (2*y + (x floordiv 2) + 3, 4*y + 3*(x floordiv 2) + 5)"},
       {"(x, y) : (y >= 0, -x - 1 >= 0)",
        "(x, y) -> (-x + 2*y + 3, -3*x + 4*y + 5)"}});
  EXPECT_THAT(*divPWMAF.valueAt({4, 3}), ElementsAre(11, 23));
  EXPECT_THAT(*divPWMAF.valueAt({4, -3}), ElementsAre(-1, -1));
  EXPECT_FALSE(divPWMAF.valueAt({3, 3}).has_value());
  EXPECT_FALSE(divPWMAF.valueAt({3, -3}).has_value());

  EXPECT_THAT(*divPWMAF.valueAt({-2, 3}), ElementsAre(11, 23));
  EXPECT_FALSE(divPWMAF.valueAt({-2, -3}).has_value());
}

TEST(PWMAFunction, removeIdRangeRegressionTest) {
  PWMAFunction pwmafA = parsePWMAF({
      {"(x, y) : (x == 0, y == 0, x - 2*(x floordiv 2) == 0, y - 2*(y floordiv "
       "2) == 0)",
       "(x, y) -> (0, 0)"},
  });
  PWMAFunction pwmafB = parsePWMAF({
      {"(x, y) : (x - 11*y == 0, 11*x - y == 0, x - 2*(x floordiv 2) == 0, "
       "y - 2*(y floordiv 2) == 0)",
       "(x, y) -> (0, 0)"},
  });
  EXPECT_TRUE(pwmafA.isEqual(pwmafB));
}

TEST(PWMAFunction, eliminateRedundantLocalIdRegressionTest) {
  PWMAFunction pwmafA = parsePWMAF({
      {"(x, y) : (x - 2*(x floordiv 2) == 0, x - 2*y == 0)", "(x, y) -> (y)"},
  });
  PWMAFunction pwmafB = parsePWMAF({
      {"(x, y) : (x - 2*(x floordiv 2) == 0, x - 2*y == 0)",
       "(x, y) -> (x - y)"},
  });
  EXPECT_TRUE(pwmafA.isEqual(pwmafB));
}

TEST(PWMAFunction, unionLexMaxSimple) {
  // func2 is better than func1, but func2's domain is empty.
  {
    PWMAFunction func1 = parsePWMAF({
        {"(x) : ()", "(x) -> (1)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x) : (1 == 0)", "(x) -> (2)"},
    });

    EXPECT_TRUE(func1.unionLexMax(func2).isEqual(func1));
    EXPECT_TRUE(func2.unionLexMax(func1).isEqual(func1));
  }

  // func2 is better than func1 on a subset of func1.
  {
    PWMAFunction func1 = parsePWMAF({
        {"(x) : ()", "(x) -> (1)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x) : (x >= 0, 10 - x >= 0)", "(x) -> (2)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(x) : (-1 - x >= 0)", "(x) -> (1)"},
        {"(x) : (x >= 0, 10 - x >= 0)", "(x) -> (2)"},
        {"(x) : (x - 11 >= 0)", "(x) -> (1)"},
    });

    EXPECT_TRUE(func1.unionLexMax(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMax(func1).isEqual(result));
  }

  // func1 and func2 are defined over the whole domain with different outputs.
  {
    PWMAFunction func1 = parsePWMAF({
        {"(x) : ()", "(x) -> (x)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x) : ()", "(x) -> (-x)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(x) : (x >= 0)", "(x) -> (x)"},
        {"(x) : (-1 - x >= 0)", "(x) -> (-x)"},
    });

    EXPECT_TRUE(func1.unionLexMax(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMax(func1).isEqual(result));
  }

  // func1 and func2 have disjoint domains.
  {
    PWMAFunction func1 = parsePWMAF({
        {"(x) : (x >= 0, 10 - x >= 0)", "(x) -> (1)"},
        {"(x) : (x - 71 >= 0, 80 - x >= 0)", "(x) -> (1)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x) : (x - 20 >= 0, 41 - x >= 0)", "(x) -> (2)"},
        {"(x) : (x - 101 >= 0, 120 - x >= 0)", "(x) -> (2)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(x) : (x >= 0, 10 - x >= 0)", "(x) -> (1)"},
        {"(x) : (x - 71 >= 0, 80 - x >= 0)", "(x) -> (1)"},
        {"(x) : (x - 20 >= 0, 41 - x >= 0)", "(x) -> (2)"},
        {"(x) : (x - 101 >= 0, 120 - x >= 0)", "(x) -> (2)"},
    });

    EXPECT_TRUE(func1.unionLexMin(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMin(func1).isEqual(result));
  }
}

TEST(PWMAFunction, unionLexMinSimple) {
  // func2 is better than func1, but func2's domain is empty.
  {
    PWMAFunction func1 = parsePWMAF({
        {"(x) : ()", "(x) -> (-1)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x) : (1 == 0)", "(x) -> (-2)"},
    });

    EXPECT_TRUE(func1.unionLexMin(func2).isEqual(func1));
    EXPECT_TRUE(func2.unionLexMin(func1).isEqual(func1));
  }

  // func2 is better than func1 on a subset of func1.
  {
    PWMAFunction func1 = parsePWMAF({
        {"(x) : ()", "(x) -> (-1)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x) : (x >= 0, 10 - x >= 0)", "(x) -> (-2)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(x) : (-1 - x >= 0)", "(x) -> (-1)"},
        {"(x) : (x >= 0, 10 - x >= 0)", "(x) -> (-2)"},
        {"(x) : (x - 11 >= 0)", "(x) -> (-1)"},
    });

    EXPECT_TRUE(func1.unionLexMin(func2).isEqual(result));
    EXPECT_TRUE(func2.unionLexMin(func1).isEqual(result));
  }

  // func1 and func2 are defined over the whole domain with different outputs.
  {
    PWMAFunction func1 = parsePWMAF({
        {"(x) : ()", "(x) -> (-x)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x) : ()", "(x) -> (x)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(x) : (x >= 0)", "(x) -> (-x)"},
        {"(x) : (-1 - x >= 0)", "(x) -> (x)"},
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
    PWMAFunction func1 = parsePWMAF({
        {"(x, y) : (x >= 10)", "(x, y) -> (x + y)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x, y) : (x <= 20)", "(x, y) -> (x - y)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(x, y) : (x >= 10, x <= 20, y >= 1)", "(x, y) -> (x + y)"},
        {"(x, y) : (x >= 21)", "(x, y) -> (x + y)"},
        {"(x, y) : (x <= 9)", "(x, y) -> (x - y)"},
        {"(x, y) : (x >= 10, x <= 20, y <= 0)", "(x, y) -> (x - y)"},
    });

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
    PWMAFunction func1 = parsePWMAF({
        {"(x, y) : (x >= 0, y >= 0)", "(x, y) -> (x + y, -2*x + 4)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x, y) : (x >= 0, y >= 0)", "(x, y) -> (x, 2*x - 2)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(x, y) : (x >= 0, y >= 1)", "(x, y) -> (x + y, -2*x + 4)"},
        {"(x, y) : (x >= 0, x <= 1, y == 0)", "(x, y) -> (x + y, -2*x + 4)"},
        {"(x, y) : (x >= 2, y == 0)", "(x, y) -> (x, 2*x - 2)"},
    });

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
    PWMAFunction func1 = parsePWMAF({
        {"(a, b, c) : (a >= 0, 1 - a >= 0, b >= 0, 1 - b >= 0, c "
         ">= 0, 1 - c >= 0)",
         "(a, b, c) -> (0, b, 0)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(a, b, c) : (a >= 0, 1 - a >= 0, b >= 0, 1 - b >= 0, c >= 0, 1 - "
         "c >= 0)",
         "(a, b, c) -> (a, 0, c)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(a, b, c) : (a - 1 == 0, b >= 0, 1 - b >= 0, c >= 0, 1 - c >= 0)",
         "(a, b, c) -> (a, 0, c)"},
        {"(a, b, c) : (a == 0, b - 1 == 0, c >= 0, 1 - c >= 0)",
         "(a, b, c) -> (0, b, 0)"},
        {"(a, b, c) : (a == 0, b == 0, c >= 0, 1 - c >= 0)",
         "(a, b, c) -> (a, 0, c)"},
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
  PWMAFunction func1 = parsePWMAF({
      {"(x, y) : (x >= 0, x <= 1, y >= 0, y <= 1)", "(x, y) -> (-x, y)"},
  });

  PWMAFunction func2 = parsePWMAF({
      {"(x, y) : (x >= 0, x <= 1, y >= 0, y <= 1)", "(x, y) -> (0, y - 1)"},
  });

  PWMAFunction result = parsePWMAF({
      {"(x, y) : (x == 1, y >= 0, y <= 1)", "(x, y) -> (-x, y)"},
      {"(x, y) : (x == 0, y >= 0, y <= 1)", "(x, y) -> (0, y - 1)"},
  });

  EXPECT_TRUE(func1.unionLexMin(func2).isEqual(result));
  EXPECT_TRUE(func2.unionLexMin(func1).isEqual(result));
}

TEST(PWMAFunction, unionLexMinWithDivs) {
  {
    PWMAFunction func1 = parsePWMAF({
        {"(x, y) : (x mod 5 == 0)", "(x, y) -> (x, 1)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x, y) : (x mod 7 == 0)", "(x, y) -> (x + y, 2)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(x, y) : (x mod 5 == 0, x mod 7 >= 1)", "(x, y) -> (x, 1)"},
        {"(x, y) : (x mod 7 == 0, x mod 5 >= 1)", "(x, y) -> (x + y, 2)"},
        {"(x, y) : (x mod 5 == 0, x mod 7 == 0, y >= 0)", "(x, y) -> (x, 1)"},
        {"(x, y) : (x mod 7 == 0, x mod 5 == 0, y <= -1)",
         "(x, y) -> (x + y, 2)"},
    });

    EXPECT_TRUE(func1.unionLexMin(func2).isEqual(result));
  }

  {
    PWMAFunction func1 = parsePWMAF({
        {"(x) : (x >= 0, x <= 1000)", "(x) -> (x floordiv 16)"},
    });

    PWMAFunction func2 = parsePWMAF({
        {"(x) : (x >= 0, x <= 1000)", "(x) -> ((x + 10) floordiv 17)"},
    });

    PWMAFunction result = parsePWMAF({
        {"(x) : (x >= 0, x <= 1000, x floordiv 16 <= (x + 10) floordiv 17)",
         "(x) -> (x floordiv 16)"},
        {"(x) : (x >= 0, x <= 1000, x floordiv 16 >= (x + 10) floordiv 17 + 1)",
         "(x) -> ((x + 10) floordiv 17)"},
    });

    EXPECT_TRUE(func1.unionLexMin(func2).isEqual(result));
  }
}
