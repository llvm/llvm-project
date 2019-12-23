//===- RulesTest.cpp - Rules unit tests -----------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Quantizer/Support/Rules.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::quantizer;

namespace {

using TestDiscreteFact = DiscreteFact<int>;

TEST(ExpandingMinMaxReducer, Basic) {
  ExpandingMinMaxFact f;
  EXPECT_FALSE(f.hasValue());

  // First assertion always modifies.
  EXPECT_TRUE(modified(f.assertValue(0, {-1.0, 1.0})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-1.0, 1.0), f.getValue());
  EXPECT_EQ(0, f.getSalience());

  // Assertion in the same band expands.
  EXPECT_TRUE(modified(f.assertValue(0, {-0.5, 2.0})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-1.0, 2.0), f.getValue());
  EXPECT_EQ(0, f.getSalience());

  EXPECT_TRUE(modified(f.assertValue(0, {-2.0, 0.5})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-2.0, 2.0), f.getValue());
  EXPECT_EQ(0, f.getSalience());

  // Same band smaller bound does not modify.
  EXPECT_FALSE(modified(f.assertValue(0, {-0.5, 0.5})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-2.0, 2.0), f.getValue());
  EXPECT_EQ(0, f.getSalience());

  // Higher salience overrides.
  EXPECT_TRUE(modified(f.assertValue(10, {-0.2, 0.2})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-0.2, 0.2), f.getValue());
  EXPECT_EQ(10, f.getSalience());

  // Lower salience no-ops.
  EXPECT_FALSE(modified(f.assertValue(5, {-2.0, 2.0})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-0.2, 0.2), f.getValue());
  EXPECT_EQ(10, f.getSalience());

  // Merge from a fact without a value no-ops.
  ExpandingMinMaxFact f1;
  EXPECT_FALSE(modified(f.mergeFrom(f1)));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-0.2, 0.2), f.getValue());
  EXPECT_EQ(10, f.getSalience());

  // Merge from a fact with a value merges.
  EXPECT_TRUE(modified(f1.mergeFrom(f)));
  EXPECT_TRUE(f1.hasValue());
  EXPECT_EQ(std::make_pair(-0.2, 0.2), f1.getValue());
  EXPECT_EQ(10, f1.getSalience());
}

TEST(TestDiscreteFact, Basic) {
  TestDiscreteFact f;
  EXPECT_FALSE(f.hasValue());

  // Initial value.
  EXPECT_TRUE(modified(f.assertValue(0, {2})));
  EXPECT_FALSE(modified(f.assertValue(0, {2})));
  EXPECT_EQ(2, f.getValue().value);
  EXPECT_FALSE(f.getValue().conflict);

  // Conflicting update.
  EXPECT_TRUE(modified(f.assertValue(0, {4})));
  EXPECT_EQ(2, f.getValue().value); // Arbitrary but known to be first wins.
  EXPECT_TRUE(f.getValue().conflict);

  // Further update still conflicts.
  EXPECT_FALSE(modified(f.assertValue(0, {4})));
  EXPECT_EQ(2, f.getValue().value); // Arbitrary but known to be first wins.
  EXPECT_TRUE(f.getValue().conflict);

  // Different salience update does not conflict.
  EXPECT_TRUE(modified(f.assertValue(1, {6})));
  EXPECT_EQ(6, f.getValue().value);
  EXPECT_FALSE(f.getValue().conflict);
}

} // end anonymous namespace
