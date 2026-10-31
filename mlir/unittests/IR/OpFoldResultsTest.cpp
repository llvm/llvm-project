//===- OpFoldResultsTest.cpp - OpFoldResults unit tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpFoldResult.h"
#include "mlir/IR/Operation.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
class OpFoldResultsTest : public ::testing::Test {
protected:
  OpFoldResultsTest() : builder(&context) {
    context.allowUnregisteredDialects();
    i32 = builder.getI32Type();
    f32 = builder.getF32Type();
  }

  ~OpFoldResultsTest() override {
    // Destroy users before the ops that define their operands.
    for (Operation *op : llvm::reverse(ops))
      op->destroy();
  }

  /// Create an op with the given result types and operands. The fixture
  /// destroys it.
  Operation *createOp(TypeRange resultTypes, StringRef name = "foo.bar",
                      ValueRange operands = {}) {
    OperationState state(UnknownLoc::get(&context), name);
    state.addTypes(resultTypes);
    state.addOperands(operands);
    Operation *op = Operation::create(state);
    ops.push_back(op);
    return op;
  }

  MLIRContext context;
  Builder builder;
  Type i32;
  Type f32;
  SmallVector<Operation *> ops;
};
} // namespace

/// Normalize `results`, a fold result of `op`.
static NormalizedOpFoldResults normalize(Operation *op, OpFoldResults results) {
  return NormalizedOpFoldResults(op, std::move(results));
}

/// Build the replacements of a fold.
static SmallVector<OpFoldResult>
replacements(std::initializer_list<OpFoldResult> list) {
  return SmallVector<OpFoldResult>(list);
}

TEST_F(OpFoldResultsTest, DefaultIsFailure) {
  NormalizedOpFoldResults result;
  EXPECT_TRUE(failed(result));
  EXPECT_FALSE(result.modifiedInPlace());
  EXPECT_FALSE(result.replacesAny());
  EXPECT_FALSE(result.replacesAll());
  EXPECT_TRUE(result.getReplacements().empty());
}

TEST_F(OpFoldResultsTest, LogicalResult) {
  Operation *op = createOp({i32});

  NormalizedOpFoldResults inPlace = normalize(op, success());
  EXPECT_TRUE(succeeded(inPlace));
  EXPECT_TRUE(inPlace.modifiedInPlace());
  EXPECT_FALSE(inPlace.replacesAny());

  NormalizedOpFoldResults failedResult = normalize(op, failure());
  EXPECT_TRUE(failed(failedResult));
  EXPECT_FALSE(failedResult.modifiedInPlace());
}

TEST_F(OpFoldResultsTest, Range) {
  Operation *producer = createOp({i32, i32});
  Operation *op = createOp({i32, i32});
  Attribute attr = builder.getI32IntegerAttr(1);

  NormalizedOpFoldResults fromVector =
      normalize(op, replacements({producer->getResult(0), attr}));
  EXPECT_TRUE(succeeded(fromVector));
  EXPECT_FALSE(fromVector.modifiedInPlace());
  EXPECT_TRUE(fromVector.replacesAll());
  ASSERT_EQ(fromVector.getReplacements().size(), 2u);
  EXPECT_EQ(fromVector.getReplacements()[0],
            OpFoldResult(producer->getResult(0)));
  EXPECT_EQ(fromVector.getReplacements()[1], OpFoldResult(attr));

  NormalizedOpFoldResults fromRange = normalize(op, producer->getResults());
  EXPECT_TRUE(fromRange.replacesAll());

  // An empty range is a failure.
  EXPECT_TRUE(failed(normalize(op, ValueRange())));
  EXPECT_TRUE(failed(normalize(op, SmallVector<OpFoldResult>())));
}

TEST_F(OpFoldResultsTest, NormalizeMapsOwnResultsToKeep) {
  Operation *op = createOp({i32, i32});
  Attribute attr = builder.getI32IntegerAttr(4);

  NormalizedOpFoldResults result =
      normalize(op, replacements({op->getResult(0), attr}));
  EXPECT_TRUE(succeeded(result));
  EXPECT_TRUE(result.replacesAny());
  EXPECT_FALSE(result.replacesAll());
  ASSERT_EQ(result.getReplacements().size(), 2u);
  EXPECT_FALSE(result.getReplacements()[0]);
  EXPECT_EQ(result.getReplacements()[1], OpFoldResult(attr));

  // A replacement may name another result of the op if that result is kept.
  NormalizedOpFoldResults forward =
      normalize(op, replacements({op->getResult(1), OpFoldResult()}));
  EXPECT_TRUE(succeeded(forward));
  ASSERT_EQ(forward.getReplacements().size(), 2u);
  EXPECT_EQ(forward.getReplacements()[0], OpFoldResult(op->getResult(1)));
  EXPECT_FALSE(forward.getReplacements()[1]);
}

TEST_F(OpFoldResultsTest, NormalizeCollapsesToFailure) {
  Operation *op = createOp({i32, i32});

  NormalizedOpFoldResults ownResults = normalize(op, op->getResults());
  EXPECT_TRUE(failed(ownResults));
  EXPECT_FALSE(ownResults.replacesAny());
  EXPECT_TRUE(ownResults.getReplacements().empty());

  NormalizedOpFoldResults nulls =
      normalize(op, replacements({OpFoldResult(), OpFoldResult()}));
  EXPECT_TRUE(failed(nulls));
  EXPECT_TRUE(nulls.getReplacements().empty());
}

TEST_F(OpFoldResultsTest, InPlaceBit) {
  Operation *op = createOp({i32, i32});
  Attribute attr = builder.getI32IntegerAttr(2);

  // An in-place change keeps a result that replaces nothing successful.
  NormalizedOpFoldResults inPlace = normalize(op, op->getResults());
  inPlace.setModifiedInPlace();
  EXPECT_TRUE(succeeded(inPlace));
  EXPECT_TRUE(inPlace.modifiedInPlace());
  EXPECT_FALSE(inPlace.replacesAny());

  // The bit is independent of the replacements.
  NormalizedOpFoldResults partial =
      normalize(op, replacements({OpFoldResult(), attr}));
  partial.setModifiedInPlace();
  EXPECT_TRUE(partial.modifiedInPlace());
  EXPECT_TRUE(partial.replacesAny());
  partial.setModifiedInPlace(false);
  EXPECT_FALSE(partial.modifiedInPlace());
  EXPECT_TRUE(succeeded(partial));
}

TEST_F(OpFoldResultsTest, ZeroResultInPlaceDoesNotReplaceAll) {
  Operation *op = createOp({});

  NormalizedOpFoldResults inPlace = normalize(op, success());
  EXPECT_TRUE(succeeded(inPlace));
  EXPECT_FALSE(inPlace.replacesAny());
  EXPECT_FALSE(inPlace.replacesAll());

  EXPECT_FALSE(normalize(op, failure()).replacesAll());
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
namespace {
class OpFoldResultsDeathTest : public OpFoldResultsTest {};
} // namespace

TEST_F(OpFoldResultsDeathTest, ValueReplacementOfIncorrectType) {
  Operation *producer = createOp({f32});
  Operation *op = createOp({i32, i32});
  EXPECT_DEATH((void)normalize(
                   op, replacements({producer->getResult(0), OpFoldResult()})),
               "incorrect fold result type");
}

TEST_F(OpFoldResultsDeathTest, ReplacementCountMismatch) {
  Operation *op = createOp({i32, i32});
  Attribute attr = builder.getI32IntegerAttr(1);
  EXPECT_DEATH((void)normalize(op, replacements({attr, attr, attr})),
               "expected one replacement per operation result");
}

#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST
