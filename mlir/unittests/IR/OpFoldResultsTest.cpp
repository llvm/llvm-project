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

static void expectOneReplacement(const OpFoldResults &result,
                                 OpFoldResult expected) {
  EXPECT_TRUE(result.succeeded());
  EXPECT_FALSE(result.modifiedInPlace());
  EXPECT_TRUE(result.replacesAny());
  EXPECT_TRUE(result.replacesAll());
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], expected);
}

// The helpers below return the way a folder returns, so they use copy
// initialization.
static OpFoldResults foldToEmptyBraces() { return {}; }
static OpFoldResults foldToTypedValue(TypedValue<IntegerType> value) {
  return value;
}
static OpFoldResults foldToIntegerAttr(IntegerAttr attr) { return attr; }
static OpFoldResults foldToArrayAttr(ArrayAttr attr) { return attr; }
static OpFoldResults foldToList(OpFoldResult lhs, OpFoldResult rhs) {
  return {lhs, rhs};
}

TEST_F(OpFoldResultsTest, DefaultIsFailure) {
  OpFoldResults result;
  EXPECT_TRUE(result.failed());
  EXPECT_FALSE(result.succeeded());
  EXPECT_FALSE(result.modifiedInPlace());
  EXPECT_FALSE(result.replacesAny());
  EXPECT_FALSE(result.replacesAll());
  EXPECT_EQ(result.size(), 0u);
  EXPECT_TRUE(result.getReplacements().empty());
}

TEST_F(OpFoldResultsTest, EmptyBracesAreFailure) {
  OpFoldResults result = foldToEmptyBraces();
  EXPECT_TRUE(result.failed());
  EXPECT_FALSE(result.modifiedInPlace());
  EXPECT_EQ(result.size(), 0u);
}

TEST_F(OpFoldResultsTest, NullptrIsFailure) {
  OpFoldResults result = nullptr;
  EXPECT_TRUE(result.failed());
  EXPECT_FALSE(result.modifiedInPlace());
  EXPECT_EQ(result.size(), 0u);
}

TEST_F(OpFoldResultsTest, LogicalResult) {
  OpFoldResults inPlace = success();
  EXPECT_TRUE(inPlace.succeeded());
  EXPECT_TRUE(inPlace.modifiedInPlace());
  EXPECT_FALSE(inPlace.replacesAny());
  EXPECT_FALSE(inPlace.replacesAll());
  EXPECT_EQ(inPlace.size(), 0u);

  OpFoldResults failedResult = failure();
  EXPECT_TRUE(failedResult.failed());
  EXPECT_FALSE(failedResult.modifiedInPlace());
  EXPECT_FALSE(failedResult.replacesAny());
  EXPECT_EQ(failedResult.size(), 0u);
}

TEST_F(OpFoldResultsTest, SingleReplacement) {
  Operation *producer = createOp({i32});
  Operation *op = createOp({i32});
  Value value = producer->getResult(0);
  IntegerAttr intAttr = builder.getI32IntegerAttr(7);
  Attribute attr = intAttr;

  OpFoldResults fromOpFoldResult = OpFoldResult(attr);
  expectOneReplacement(fromOpFoldResult, attr);

  OpFoldResults fromValue = value;
  expectOneReplacement(fromValue, value);

  OpFoldResults fromAttr = attr;
  expectOneReplacement(fromAttr, attr);

  // TypedValue and IntegerAttr use the Value and Attribute constructors.
  OpFoldResults fromTypedValue =
      foldToTypedValue(cast<TypedValue<IntegerType>>(value));
  expectOneReplacement(fromTypedValue, value);

  OpFoldResults fromIntegerAttr = foldToIntegerAttr(intAttr);
  expectOneReplacement(fromIntegerAttr, attr);

  // The object holds one replacement for the single result.
  fromValue.normalize(op);
  expectOneReplacement(fromValue, value);
  EXPECT_EQ(fromValue.size(), op->getNumResults());
}

TEST_F(OpFoldResultsTest, ArrayAttrFillsOneReplacement) {
  Operation *op = createOp({i32});
  ArrayAttr arrayAttr = builder.getI32ArrayAttr({1, 2, 3});
  OpFoldResults result = foldToArrayAttr(arrayAttr);
  expectOneReplacement(result, arrayAttr);
  result.normalize(op);
  expectOneReplacement(result, arrayAttr);
}

TEST_F(OpFoldResultsTest, InitializerList) {
  Operation *producer = createOp({i32});
  Operation *op = createOp({i32, f32});
  Value value = producer->getResult(0);
  Attribute attr = builder.getF32FloatAttr(1.0);

  OpFoldResults result = {value, attr};
  EXPECT_TRUE(result.succeeded());
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], OpFoldResult(value));
  EXPECT_EQ(result[1], OpFoldResult(attr));
  result.normalize(op);
  EXPECT_TRUE(result.succeeded());
  EXPECT_FALSE(result.modifiedInPlace());
  EXPECT_TRUE(result.replacesAll());

  OpFoldResults partial = foldToList(OpFoldResult(), attr);
  partial.normalize(op);
  EXPECT_TRUE(partial.succeeded());
  EXPECT_TRUE(partial.replacesAny());
  EXPECT_FALSE(partial.replacesAll());
  ASSERT_EQ(partial.size(), 2u);
  EXPECT_FALSE(partial[0]);
  EXPECT_EQ(partial[1], OpFoldResult(attr));
}

TEST_F(OpFoldResultsTest, Range) {
  Operation *producer = createOp({i32, f32});
  Operation *op = createOp({i32, f32});
  Attribute attr = builder.getI32IntegerAttr(3);

  SmallVector<OpFoldResult> vector = {attr, OpFoldResult()};
  OpFoldResults fromVector = vector;
  EXPECT_TRUE(fromVector.succeeded());
  ASSERT_EQ(fromVector.size(), 2u);
  EXPECT_EQ(fromVector[0], OpFoldResult(attr));
  EXPECT_FALSE(fromVector[1]);
  fromVector.normalize(op);
  EXPECT_TRUE(fromVector.replacesAny());
  EXPECT_FALSE(fromVector.replacesAll());

  ValueRange range = producer->getResults();
  OpFoldResults fromRange = range;
  EXPECT_TRUE(fromRange.succeeded());
  ASSERT_EQ(fromRange.size(), 2u);
  EXPECT_EQ(fromRange[0], OpFoldResult(producer->getResult(0)));
  EXPECT_EQ(fromRange[1], OpFoldResult(producer->getResult(1)));
  fromRange.normalize(op);
  EXPECT_TRUE(fromRange.replacesAll());

  OpFoldResults fromEmptyRange = ValueRange();
  EXPECT_TRUE(fromEmptyRange.failed());
  EXPECT_EQ(fromEmptyRange.size(), 0u);

  OpFoldResults fromEmptyVector = SmallVector<OpFoldResult>();
  EXPECT_TRUE(fromEmptyVector.failed());

  SmallVector<OpFoldResult> nulls(2);
  OpFoldResults fromNulls = nulls;
  EXPECT_TRUE(fromNulls.failed());
}

TEST_F(OpFoldResultsTest, IncrementalForm) {
  Operation *producer = createOp({i32});
  Operation *op = createOp({i32, i32});
  Value value = producer->getResult(0);
  Attribute attr = builder.getI32IntegerAttr(5);

  OpFoldResults result(op);
  EXPECT_TRUE(result.failed());
  EXPECT_FALSE(result.modifiedInPlace());
  EXPECT_FALSE(result.replacesAny());
  EXPECT_EQ(result.size(), 2u);

  result.replace(op->getResult(1), attr);
  EXPECT_TRUE(result.succeeded());
  EXPECT_TRUE(result.replacesAny());
  EXPECT_FALSE(result.replacesAll());
  EXPECT_FALSE(result[0]);
  EXPECT_EQ(result[1], OpFoldResult(attr));

  // A literal 0 picks the unsigned overload.
  result.replace(0, value);
  EXPECT_TRUE(result.replacesAll());
  EXPECT_EQ(result[0], OpFoldResult(value));

  // The last write wins.
  result.replace(0u, attr);
  EXPECT_EQ(result[0], OpFoldResult(attr));

  // A null replacement, or the result itself, keeps the result.
  result.replace(op->getResult(0), OpFoldResult());
  EXPECT_FALSE(result[0]);
  result.replace(op->getResult(1), op->getResult(1));
  EXPECT_FALSE(result[1]);
  EXPECT_TRUE(result.failed());
  EXPECT_FALSE(result.replacesAny());

  result.setModifiedInPlace();
  EXPECT_TRUE(result.succeeded());
  EXPECT_TRUE(result.modifiedInPlace());
  result.setModifiedInPlace(false);
  EXPECT_TRUE(result.failed());
  EXPECT_FALSE(result.modifiedInPlace());

  // normalize() maps the op's own result, set by index, to "keep".
  OpFoldResults ownResult(op);
  ownResult.replace(1u, op->getResult(1));
  ownResult.normalize(op);
  EXPECT_TRUE(ownResult.failed());
  EXPECT_EQ(ownResult.size(), 0u);

  OpFoldResults partial(op);
  partial.replace(1u, value);
  partial.normalize(op);
  EXPECT_TRUE(partial.succeeded());
  ASSERT_EQ(partial.size(), 2u);
  EXPECT_FALSE(partial[0]);
  EXPECT_EQ(partial[1], OpFoldResult(value));
  ArrayRef<OpFoldResult> replacements = partial.getReplacements();
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_FALSE(replacements[0]);
  EXPECT_EQ(replacements[1], OpFoldResult(value));
}

TEST_F(OpFoldResultsTest, NullSafety) {
  Operation *op = createOp({i32, i32});

  EXPECT_TRUE(OpFoldResults(Value()).failed());
  EXPECT_TRUE(OpFoldResults(Attribute()).failed());
  EXPECT_TRUE(OpFoldResults(OpFoldResult()).failed());

  OpFoldResults nullList = {OpFoldResult(), OpFoldResult()};
  EXPECT_TRUE(nullList.failed());
  nullList.normalize(op);
  EXPECT_TRUE(nullList.failed());
  EXPECT_EQ(nullList.size(), 0u);
  EXPECT_FALSE(nullList[0]);
  EXPECT_FALSE(nullList[1]);

  Attribute attr = builder.getI32IntegerAttr(1);
  OpFoldResults mixed = {OpFoldResult(), attr};
  mixed.normalize(op);
  EXPECT_TRUE(mixed.succeeded());
  EXPECT_FALSE(mixed[0]);
  EXPECT_EQ(mixed[1], OpFoldResult(attr));

  OpFoldResults incremental(op);
  incremental.replace(0u, OpFoldResult());
  incremental.normalize(op);
  EXPECT_TRUE(incremental.failed());
  EXPECT_EQ(incremental.size(), 0u);
}

TEST_F(OpFoldResultsTest, NormalizeMapsOwnResultsToKeep) {
  Operation *op = createOp({i32, i32});
  Attribute attr = builder.getI32IntegerAttr(4);

  OpFoldResults result = {op->getResult(0), attr};
  result.normalize(op);
  EXPECT_TRUE(result.succeeded());
  EXPECT_TRUE(result.replacesAny());
  EXPECT_FALSE(result.replacesAll());
  ASSERT_EQ(result.size(), 2u);
  EXPECT_FALSE(result[0]);
  EXPECT_EQ(result[1], OpFoldResult(attr));

  // A replacement may name another result of the op if that result is kept.
  OpFoldResults forward = {op->getResult(1), OpFoldResult()};
  forward.normalize(op);
  EXPECT_TRUE(forward.succeeded());
  ASSERT_EQ(forward.size(), 2u);
  EXPECT_EQ(forward[0], OpFoldResult(op->getResult(1)));
  EXPECT_FALSE(forward[1]);
}

TEST_F(OpFoldResultsTest, NormalizeCollapsesToFailure) {
  Operation *op = createOp({i32, i32});
  Operation *oneResultOp = createOp({i32});

  OpFoldResults ownResults = {op->getResult(0), op->getResult(1)};
  ownResults.normalize(op);
  EXPECT_TRUE(ownResults.failed());
  EXPECT_FALSE(ownResults.replacesAny());
  EXPECT_EQ(ownResults.size(), 0u);
  EXPECT_TRUE(ownResults.getReplacements().empty());

  OpFoldResults ownRange = op->getResults();
  ownRange.normalize(op);
  EXPECT_TRUE(ownRange.failed());

  OpFoldResults ownResult = oneResultOp->getResult(0);
  ownResult.normalize(oneResultOp);
  EXPECT_TRUE(ownResult.failed());

  OpFoldResults nothingReplaced(op);
  nothingReplaced.normalize(op);
  EXPECT_TRUE(nothingReplaced.failed());
  EXPECT_EQ(nothingReplaced.size(), 0u);

  // An in-place change keeps the result successful.
  OpFoldResults inPlace = {op->getResult(0), op->getResult(1)};
  inPlace.setModifiedInPlace();
  inPlace.normalize(op);
  EXPECT_TRUE(inPlace.succeeded());
  EXPECT_TRUE(inPlace.modifiedInPlace());
  EXPECT_FALSE(inPlace.replacesAny());
  EXPECT_FALSE(inPlace.replacesAll());
  EXPECT_EQ(inPlace.size(), 0u);
}

TEST_F(OpFoldResultsTest, NormalizeIsIdempotent) {
  Operation *op = createOp({i32, i32});
  Attribute attr = builder.getI32IntegerAttr(6);

  OpFoldResults result = {op->getResult(0), attr};
  result.setModifiedInPlace();
  result.normalize(op);
  SmallVector<OpFoldResult> replacements(result.getReplacements());
  result.normalize(op);
  EXPECT_TRUE(result.succeeded());
  EXPECT_TRUE(result.modifiedInPlace());
  EXPECT_TRUE(result.replacesAny());
  EXPECT_FALSE(result.replacesAll());
  ASSERT_EQ(result.size(), replacements.size());
  for (unsigned i = 0, e = replacements.size(); i < e; ++i)
    EXPECT_EQ(result[i], replacements[i]);

  OpFoldResults failedResult = failure();
  failedResult.normalize(op);
  failedResult.normalize(op);
  EXPECT_TRUE(failedResult.failed());
  EXPECT_EQ(failedResult.size(), 0u);

  OpFoldResults inPlace = success();
  inPlace.normalize(op);
  inPlace.normalize(op);
  EXPECT_TRUE(inPlace.succeeded());
  EXPECT_TRUE(inPlace.modifiedInPlace());
  EXPECT_EQ(inPlace.size(), 0u);
}

TEST_F(OpFoldResultsTest, ZeroResultInPlaceDoesNotReplaceAll) {
  Operation *op = createOp({});

  OpFoldResults inPlace = success();
  inPlace.normalize(op);
  EXPECT_TRUE(inPlace.succeeded());
  EXPECT_TRUE(inPlace.modifiedInPlace());
  EXPECT_FALSE(inPlace.replacesAny());
  EXPECT_FALSE(inPlace.replacesAll());
  EXPECT_EQ(inPlace.size(), 0u);

  OpFoldResults failedResult = failure();
  failedResult.normalize(op);
  EXPECT_TRUE(failedResult.failed());
  EXPECT_FALSE(failedResult.replacesAll());
}

TEST_F(OpFoldResultsTest, FreeHelpersMatchMembers) {
  Operation *op = createOp({i32, i32});
  OpFoldResults partial(op);
  partial.replace(1u, builder.getI32IntegerAttr(1));
  OpFoldResults inPlace = success();
  OpFoldResults failedResult = failure();
  for (const OpFoldResults *result : {&partial, &inPlace, &failedResult}) {
    EXPECT_EQ(succeeded(*result), result->succeeded());
    EXPECT_EQ(failed(*result), result->failed());
  }
  EXPECT_TRUE(succeeded(partial));
  EXPECT_TRUE(succeeded(inPlace));
  EXPECT_TRUE(failed(failedResult));
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
namespace {
class OpFoldResultsDeathTest : public OpFoldResultsTest {};
} // namespace

TEST_F(OpFoldResultsDeathTest, ValueReplacementOfIncorrectType) {
  Operation *producer = createOp({f32});
  Operation *op = createOp({i32, i32});
  OpFoldResults result(op);
  result.replace(0u, producer->getResult(0));
  EXPECT_DEATH(result.normalize(op), "incorrect fold result type");
}

TEST_F(OpFoldResultsDeathTest, ReplacementCountMismatch) {
  Operation *op = createOp({i32, i32});
  Attribute attr = builder.getI32IntegerAttr(1);
  OpFoldResults result = {attr, attr, attr};
  EXPECT_DEATH(result.normalize(op),
               "expected one replacement per operation result");
}

#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST
