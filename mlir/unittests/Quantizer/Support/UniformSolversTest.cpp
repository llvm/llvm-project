//===- UniformSolversTest.cpp - Tests for uniform solvers -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Quantizer/Support/UniformSolvers.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::quantizer;

namespace {

const double kEpsilon = 1e-12;

TEST(UniformMathTest, testAsym) {
  UniformParamsFromMinMaxSolver s(UniformStorageParams::getQuint8(), -8, 8.123);
  ASSERT_TRUE(s.compute());

  llvm::errs() << "testBasic Results: " << s << "\n";

  EXPECT_EQ(0.0, s.dequantize(s.getZp())); // Exact.
  EXPECT_EQ(s.getZp(), s.quantize(0.0));
  EXPECT_GE(s.getAdjMax() + kEpsilon, s.getBoundingMax());
  EXPECT_LE(s.getAdjMin() - kEpsilon, s.getBoundingMin());
}

TEST(UniformMathTest, testPOT) {
  UniformParamsFromMinMaxSolver s(UniformStorageParams::getQuint8(), -8,
                                  7.9375);
  ASSERT_TRUE(s.compute());

  llvm::errs() << "testPOT Results: " << s << "\n";

  // POT ranges should be exact.
  EXPECT_EQ(128, s.getZp());
  EXPECT_NEAR(6.25e-2, s.getScale(), kEpsilon);
  EXPECT_EQ(0.0, s.dequantize(s.getZp())); // Exact.
  EXPECT_EQ(s.getZp(), s.quantize(0.0));
  EXPECT_GE(s.getAdjMax() + kEpsilon, s.getBoundingMax());
  EXPECT_LE(s.getAdjMin() - kEpsilon, s.getBoundingMin());
}

TEST(UniformMathTest, testLopsidedPositive) {
  UniformParamsFromMinMaxSolver s(UniformStorageParams::getQuint8(), 1.0, 8.0);
  ASSERT_TRUE(s.compute());

  llvm::errs() << "testLopsidedPositive Results: " << s << "\n";

  EXPECT_EQ(0, s.getZp());
  EXPECT_EQ(0.0, s.dequantize(s.getZp())); // Exact.
  EXPECT_EQ(0, s.quantize(0.0));
  EXPECT_GE(s.getAdjMax() + kEpsilon, s.getBoundingMax());
  EXPECT_LE(s.getAdjMin() - kEpsilon, s.getBoundingMin());
}

TEST(UniformMathTest, testLopsidedNegative) {
  UniformParamsFromMinMaxSolver s(UniformStorageParams::getQuint8(), -72.0,
                                  -4.0);
  ASSERT_TRUE(s.compute());

  llvm::errs() << "testLopsidedNegative Results: " << s << "\n";

  EXPECT_EQ(255, s.getZp());
  EXPECT_EQ(0.0, s.dequantize(s.getZp())); // Exact.
  EXPECT_EQ(255, s.quantize(0.0));
  EXPECT_GE(s.getAdjMax() + kEpsilon, s.getBoundingMax());
  EXPECT_LE(s.getAdjMin() - kEpsilon, s.getBoundingMin());
}

TEST(UniformMathTest, testLargeRange) {
  UniformParamsFromMinMaxSolver s(UniformStorageParams::getQuint8(), -123.23389,
                                  231.1289);
  ASSERT_TRUE(s.compute());

  llvm::errs() << "testLargeRange Results: " << s << "\n";

  // EXPECT_EQ(255, s.getZp());
  EXPECT_EQ(0.0, s.dequantize(s.getZp())); // Exact.
  EXPECT_EQ(s.getZp(), s.quantize(0.0));
  EXPECT_GE(s.getAdjMax() + kEpsilon, s.getBoundingMax());
  EXPECT_LE(s.getAdjMin() - kEpsilon, s.getBoundingMin());
}

TEST(UniformMathTest, test16BitLargeRange) {
  UniformParamsFromMinMaxSolver s(UniformStorageParams::getQuint16(),
                                  -123.23389, 231.1289);
  ASSERT_TRUE(s.compute());

  llvm::errs() << "test16BitLargeRange Results: " << s << "\n";

  EXPECT_EQ(0.0, s.dequantize(s.getZp())); // Exact.
  EXPECT_EQ(s.getZp(), s.quantize(0.0));
  EXPECT_GE(s.getAdjMax() + kEpsilon, s.getBoundingMax());
  EXPECT_LE(s.getAdjMin() - kEpsilon, s.getBoundingMin());
}

TEST(UniformMathTest, testQuint8SymmetricRight) {
  UniformParamsFromMinMaxSolver s(
      UniformStorageParams::getQuint8SymmetricRight(), -123.23389, 231.1289);
  ASSERT_TRUE(s.compute());

  llvm::errs() << "testQuint8SymmetricRight Results: " << s << "\n";

  EXPECT_EQ(0.0, s.dequantize(s.getZp())); // Exact.
  EXPECT_EQ(s.getZp(), s.quantize(0.0));
  EXPECT_GE(s.getAdjMax() + kEpsilon, s.getBoundingMax());
  EXPECT_LE(s.getAdjMin() - kEpsilon, s.getBoundingMin());
}

TEST(UniformMathTest, testQuint4) {
  UniformParamsFromMinMaxSolver s({15, 0}, -1.0, 1.0);
  ASSERT_TRUE(s.compute());

  llvm::errs() << "testQuint4 Results: " << s << "\n";

  EXPECT_EQ(0.0, s.dequantize(s.getZp())); // Exact.
  EXPECT_EQ(s.getZp(), s.quantize(0.0));
  EXPECT_GE(s.getAdjMax() + kEpsilon, s.getBoundingMax());
  EXPECT_LE(s.getAdjMin() - kEpsilon, s.getBoundingMin());
}

TEST(UniformMathTest, testNan) {
  UniformParamsFromMinMaxSolver s({0, 0}, -1.0, 1.0);
  ASSERT_FALSE(s.compute());
}

TEST(UniformMathTest, testBadBounds) {
  UniformParamsFromMinMaxSolver s(UniformStorageParams::getQuint16(), 123.23389,
                                  -231.1289);
  ASSERT_FALSE(s.compute());
}

TEST(UniformMathTest, testZeroBounds) {
  UniformParamsFromMinMaxSolver s(UniformStorageParams::getQuint16(), 0, 0);
  ASSERT_FALSE(s.compute());
}

} // end namespace
