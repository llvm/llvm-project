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
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpFoldResult.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "gtest/gtest.h"

#include <functional>

using namespace mlir;

// The fallback TypeID resolver rejects a trait template in an anonymous
// namespace, so the test dialect lives in a named namespace.
namespace op_fold_results_test {
using LegacyFoldFn =
    std::function<LogicalResult(Operation *, SmallVectorImpl<OpFoldResult> &)>;

/// Per-test behavior of the ops, the trait, and the dialect interface below.
struct FoldState {
  LegacyFoldFn traitFoldFn;
  LegacyFoldFn dialectFoldFn;
  unsigned traitCalls = 0;
  unsigned dialectCalls = 0;
};

/// The state of the running test. The test fixture owns it.
static FoldState *foldState = nullptr;

template <typename ConcreteType>
struct LegacyFoldTrait
    : public OpTrait::TraitBase<ConcreteType, LegacyFoldTrait> {
  static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute>,
                                 SmallVectorImpl<OpFoldResult> &results) {
    ++foldState->traitCalls;
    return foldState->traitFoldFn ? foldState->traitFoldFn(op, results)
                                  : failure();
  }
};

/// An op with two results and no fold method. Only its trait folds.
struct PartialFoldOp : public Op<PartialFoldOp, OpTrait::NResults<2>::Impl,
                                 OpTrait::VariadicOperands, LegacyFoldTrait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartialFoldOp)
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static StringRef getOperationName() { return "fold_test.partial"; }
};

struct TestFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;
  LogicalResult fold(Operation *op, ArrayRef<Attribute>,
                     SmallVectorImpl<OpFoldResult> &results) const final {
    ++foldState->dialectCalls;
    return foldState->dialectFoldFn ? foldState->dialectFoldFn(op, results)
                                    : failure();
  }
};

struct FoldTestDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldTestDialect)
  static constexpr StringLiteral getDialectNamespace() { return "fold_test"; }
  explicit FoldTestDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<FoldTestDialect>()) {
    addOperations<PartialFoldOp>();
    addInterfaces<TestFoldInterface>();
  }
};
} // namespace op_fold_results_test

using namespace op_fold_results_test;

namespace {
class OpFoldResultsTest : public ::testing::Test {
protected:
  OpFoldResultsTest() : builder(&context) {
    context.allowUnregisteredDialects();
    context.loadDialect<FoldTestDialect>();
    i32 = builder.getI32Type();
    f32 = builder.getF32Type();
    foldState = &state;
  }

  ~OpFoldResultsTest() override {
    // Destroy users before the ops that define their operands.
    for (Operation *op : llvm::reverse(ops))
      op->destroy();
    foldState = nullptr;
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

  /// Load a dynamic dialect. The fold hook of `test_fold.op` calls `foldFn`.
  /// The legacy fold hooks of `test_fold.legacy_op` and `test_fold.get_op`
  /// call `legacyFoldFn`; `test_fold.get_op` uses the legacy overload of
  /// DynamicOpDefinition::get. `test_fold.no_fold_op` has no fold hook.
  void loadDynamicDialect() {
    context.getOrLoadDynamicDialect("test_fold", [&](DynamicDialect *dialect) {
      auto verify = [](Operation *) { return success(); };
      std::unique_ptr<DynamicOpDefinition> opDef =
          DynamicOpDefinition::get("op", dialect, verify, verify);
      opDef->setFoldHookFn(
          [this](Operation *op, ArrayRef<Attribute>) { return foldFn(op); });
      dialect->registerDynamicOp(std::move(opDef));

      std::unique_ptr<DynamicOpDefinition> legacyOpDef =
          DynamicOpDefinition::get("legacy_op", dialect, verify, verify);
      legacyOpDef->setFoldHookFn(
          [this](Operation *op, ArrayRef<Attribute>,
                 SmallVectorImpl<OpFoldResult> &results) {
            return legacyFoldFn(op, results);
          });
      dialect->registerDynamicOp(std::move(legacyOpDef));

      DynamicOpDefinition::LegacyFoldHookFn legacyFold =
          [this](Operation *op, ArrayRef<Attribute>,
                 SmallVectorImpl<OpFoldResult> &results) {
            return legacyFoldFn(op, results);
          };
      dialect->registerDynamicOp(DynamicOpDefinition::get(
          "get_op", dialect, verify, verify,
          [](OpAsmParser &, OperationState &) -> ParseResult {
            return failure();
          },
          [](Operation *, OpAsmPrinter &, StringRef) {}, std::move(legacyFold),
          [](RewritePatternSet &, MLIRContext *) {},
          [](const OperationName &, NamedAttrList &) {}));

      std::unique_ptr<DynamicOpDefinition> noFoldOpDef =
          DynamicOpDefinition::get("no_fold_op", dialect, verify, verify);
      noFoldOpDef->setFoldHookFn(
          [this](Operation *op, ArrayRef<Attribute>) { return foldFn(op); });
      noFoldOpDef->setFoldHookFn(nullptr);
      dialect->registerDynamicOp(std::move(noFoldOpDef));
    });
  }

  MLIRContext context;
  Builder builder;
  Type i32;
  Type f32;
  SmallVector<Operation *> ops;
  std::function<OpFoldResults(Operation *)> foldFn;
  std::function<LogicalResult(Operation *, SmallVectorImpl<OpFoldResult> &)>
      legacyFoldFn;
  FoldState state;
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

  // The one replacement matches the single result.
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

TEST_F(OpFoldResultsTest, UnregisteredOpFoldFails) {
  Operation *op = createOp({i32});
  EXPECT_TRUE(op->fold().failed());
  SmallVector<OpFoldResult> results;
  EXPECT_TRUE(failed(op->fold(results)));
  EXPECT_TRUE(results.empty());
}

TEST_F(OpFoldResultsTest, LegacyOperationFoldKeepsStrictContract) {
  loadDynamicDialect();
  Operation *producer = createOp({i32, i32});
  Operation *op = createOp({i32, i32}, "test_fold.op");
  Attribute attr = builder.getI32IntegerAttr(1);
  SmallVector<OpFoldResult> results;

  foldFn = [](Operation *) -> OpFoldResults { return failure(); };
  EXPECT_TRUE(failed(op->fold(results)));
  EXPECT_TRUE(results.empty());
  EXPECT_TRUE(op->fold().failed());

  foldFn = [](Operation *) -> OpFoldResults { return success(); };
  EXPECT_TRUE(succeeded(op->fold(results)));
  EXPECT_TRUE(results.empty());

  foldFn = [&](Operation *) -> OpFoldResults {
    return {attr, producer->getResult(1)};
  };
  EXPECT_TRUE(succeeded(op->fold(results)));
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0], OpFoldResult(attr));
  EXPECT_EQ(results[1], OpFoldResult(producer->getResult(1)));
  results.clear();

  // The legacy overloads do not apply a partial fold. Without an in-place
  // change, the partial fold is a failure.
  foldFn = [&](Operation *foldedOp) {
    OpFoldResults partial(foldedOp);
    partial.replace(1u, attr);
    return partial;
  };
  EXPECT_TRUE(failed(op->fold(results)));
  EXPECT_TRUE(results.empty());
  OpFoldResults partialResult = op->fold();
  EXPECT_TRUE(partialResult.succeeded());
  EXPECT_FALSE(partialResult.modifiedInPlace());
  ASSERT_EQ(partialResult.size(), 2u);
  EXPECT_FALSE(partialResult[0]);
  EXPECT_EQ(partialResult[1], OpFoldResult(attr));

  // With an in-place change, the partial fold is reported as in place.
  foldFn = [&](Operation *foldedOp) {
    OpFoldResults partial(foldedOp);
    partial.replace(1u, attr);
    partial.setModifiedInPlace();
    return partial;
  };
  EXPECT_TRUE(succeeded(op->fold(results)));
  EXPECT_TRUE(results.empty());

  // A fold that keeps every result and is not in place is a failure.
  foldFn = [](Operation *foldedOp) -> OpFoldResults {
    return foldedOp->getResults();
  };
  EXPECT_TRUE(failed(op->fold(results)));
  EXPECT_TRUE(results.empty());
  EXPECT_TRUE(op->fold().failed());
}

TEST_F(OpFoldResultsTest, LegacyDynamicFoldHookKeepsStrictContract) {
  loadDynamicDialect();
  Operation *producer = createOp({i32, i32});
  Operation *op = createOp({i32, i32}, "test_fold.legacy_op");
  Attribute attr = builder.getI32IntegerAttr(1);

  legacyFoldFn = [](Operation *, SmallVectorImpl<OpFoldResult> &) {
    return failure();
  };
  EXPECT_TRUE(op->fold().failed());

  legacyFoldFn = [](Operation *, SmallVectorImpl<OpFoldResult> &) {
    return success();
  };
  OpFoldResults inPlace = op->fold();
  EXPECT_TRUE(inPlace.succeeded());
  EXPECT_TRUE(inPlace.modifiedInPlace());
  EXPECT_FALSE(inPlace.replacesAny());

  legacyFoldFn = [&](Operation *, SmallVectorImpl<OpFoldResult> &results) {
    results.push_back(attr);
    results.push_back(producer->getResult(0));
    return success();
  };
  OpFoldResults all = op->fold();
  EXPECT_TRUE(all.succeeded());
  EXPECT_FALSE(all.modifiedInPlace());
  EXPECT_TRUE(all.replacesAll());
  ASSERT_EQ(all.size(), 2u);
  EXPECT_EQ(all[0], OpFoldResult(attr));
  EXPECT_EQ(all[1], OpFoldResult(producer->getResult(0)));

  // A legacy fold that forwards the op's own results keeps every result, so
  // it is a failure.
  legacyFoldFn = [](Operation *foldedOp,
                    SmallVectorImpl<OpFoldResult> &results) {
    llvm::append_range(results, foldedOp->getResults());
    return success();
  };
  EXPECT_TRUE(op->fold().failed());
  SmallVector<OpFoldResult> results;
  EXPECT_TRUE(failed(op->fold(results)));
  EXPECT_TRUE(results.empty());
}

TEST_F(OpFoldResultsTest, LegacyDynamicFoldHookMayForwardAnotherResult) {
  loadDynamicDialect();
  Operation *op = createOp({i32, i32}, "test_fold.legacy_op");
  Attribute attr = builder.getI32IntegerAttr(1);

  // Replacement 0 is result 1, which the fold also replaces. The legacy
  // adapter does not reject this.
  legacyFoldFn = [&](Operation *foldedOp,
                     SmallVectorImpl<OpFoldResult> &results) {
    results.push_back(foldedOp->getResult(1));
    results.push_back(attr);
    return success();
  };
  OpFoldResults result = op->fold();
  EXPECT_TRUE(result.replacesAll());
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], OpFoldResult(op->getResult(1)));
  EXPECT_EQ(result[1], OpFoldResult(attr));

  SmallVector<OpFoldResult> results;
  EXPECT_TRUE(succeeded(op->fold(results)));
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0], OpFoldResult(op->getResult(1)));
  EXPECT_EQ(results[1], OpFoldResult(attr));
}

TEST_F(OpFoldResultsTest, LegacyDynamicOpDefinitionGet) {
  loadDynamicDialect();
  Operation *op = createOp({i32}, "test_fold.get_op");
  Attribute attr = builder.getI32IntegerAttr(1);

  legacyFoldFn = [&](Operation *, SmallVectorImpl<OpFoldResult> &results) {
    results.push_back(attr);
    return success();
  };
  OpFoldResults result = op->fold();
  expectOneReplacement(result, attr);

  legacyFoldFn = [](Operation *, SmallVectorImpl<OpFoldResult> &) {
    return failure();
  };
  EXPECT_TRUE(op->fold().failed());
}

TEST_F(OpFoldResultsTest, NullFoldHookFails) {
  loadDynamicDialect();
  Operation *op = createOp({i32}, "test_fold.no_fold_op");
  // The removed hook would report an in-place fold.
  foldFn = [](Operation *) -> OpFoldResults { return success(); };
  EXPECT_TRUE(op->fold().failed());
  SmallVector<OpFoldResult> results;
  EXPECT_TRUE(failed(op->fold(results)));
  EXPECT_TRUE(results.empty());
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

TEST_F(OpFoldResultsTest, DialectFoldInterfaceFallback) {
  Operation *op = createOp({i32, i32}, "fold_test.partial");
  Attribute attr = builder.getI32IntegerAttr(1);
  foldState->dialectFoldFn = [&](Operation *,
                                 SmallVectorImpl<OpFoldResult> &results) {
    results.append(2, attr);
    return success();
  };

  // The traits fail.
  OpFoldResults result = op->fold();
  EXPECT_TRUE(result.replacesAll());
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], OpFoldResult(attr));
  EXPECT_EQ(foldState->traitCalls, 1u);
  EXPECT_EQ(foldState->dialectCalls, 1u);
  SmallVector<OpFoldResult> results;
  EXPECT_TRUE(succeeded(op->fold(results)));
  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(foldState->dialectCalls, 2u);

  // The fallback result is normalized.
  foldState->dialectCalls = 0;
  foldState->dialectFoldFn = [](Operation *foldedOp,
                                SmallVectorImpl<OpFoldResult> &results) {
    llvm::append_range(results, foldedOp->getResults());
    return success();
  };
  EXPECT_TRUE(op->fold().failed());
  EXPECT_EQ(foldState->dialectCalls, 1u);
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

TEST_F(OpFoldResultsDeathTest, LegacyNullResult) {
  loadDynamicDialect();
  Operation *op = createOp({i32, i32}, "test_fold.legacy_op");
  Attribute attr = builder.getI32IntegerAttr(1);
  legacyFoldFn = [&](Operation *, SmallVectorImpl<OpFoldResult> &results) {
    results.push_back(attr);
    results.push_back(OpFoldResult());
    return success();
  };
  EXPECT_DEATH((void)op->fold(), "legacy fold returned a null result");
}
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST
