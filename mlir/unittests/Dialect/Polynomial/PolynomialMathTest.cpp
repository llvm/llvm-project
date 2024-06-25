//===- PolynomialMathTest.cpp - Polynomial math Tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::polynomial;

TEST(AddTest, checkSameDegreeAdditionOfIntPolynomial) {
  IntPolynomial x = IntPolynomial::fromCoefficients({1, 2, 3});
  IntPolynomial y = IntPolynomial::fromCoefficients({2, 3, 4});
  IntPolynomial expected = IntPolynomial::fromCoefficients({3, 5, 7});
  EXPECT_EQ(expected, x.add(y));
}

TEST(AddTest, checkDifferentDegreeAdditionOfIntPolynomial) {
  IntMonomial term2t = IntMonomial(2, 1);
  IntPolynomial x = IntPolynomial::fromMonomials({term2t}).value();
  IntPolynomial y = IntPolynomial::fromCoefficients({2, 3, 4});
  IntPolynomial expected = IntPolynomial::fromCoefficients({2, 5, 4});
  EXPECT_EQ(expected, x.add(y));
  EXPECT_EQ(expected, y.add(x));
}

TEST(AddTest, checkSameDegreeAdditionOfFloatPolynomial) {
  FloatPolynomial x = FloatPolynomial::fromCoefficients({1.5, 2.5, 3.5});
  FloatPolynomial y = FloatPolynomial::fromCoefficients({2.5, 3.5, 4.5});
  FloatPolynomial expected = FloatPolynomial::fromCoefficients({4, 6, 8});
  EXPECT_EQ(expected, x.add(y));
}

TEST(AddTest, checkDifferentDegreeAdditionOfFloatPolynomial) {
  FloatPolynomial x = FloatPolynomial::fromCoefficients({1.5, 2.5});
  FloatPolynomial y = FloatPolynomial::fromCoefficients({2.5, 3.5, 4.5});
  FloatPolynomial expected = FloatPolynomial::fromCoefficients({4, 6, 4.5});
  EXPECT_EQ(expected, x.add(y));
  EXPECT_EQ(expected, y.add(x));
}
