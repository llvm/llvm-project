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
#include "llvm/Support/Compiler.h"
#include "gtest/gtest.h"

#include <functional>

using namespace mlir;

// The fallback TypeID resolver rejects a trait instantiated with a type from
// an anonymous namespace, so the test dialect lives in a named namespace.
namespace op_fold_results_test {
/// The fold of the op below. The test fixture clears it after each test.
static std::function<OpFoldResults(Operation *)> opFoldFn;

template <typename OpT>
struct FoldAdaptorImpl {
  FoldAdaptorImpl(ArrayRef<Attribute> operands, OpT) : operands(operands) {}
  ArrayRef<Attribute> getOperands() const { return operands; }
  ArrayRef<Attribute> operands;
};

/// An op with two results that defines `OpFoldResults fold(FoldAdaptor)`.
struct PartialFoldOp : public Op<PartialFoldOp, OpTrait::NResults<2>::Impl,
                                 OpTrait::VariadicOperands> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartialFoldOp)
  using Op::Op;
  using FoldAdaptor = FoldAdaptorImpl<PartialFoldOp>;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static StringRef getOperationName() { return "fold_test.partial"; }
  OpFoldResults fold(FoldAdaptor) {
    return opFoldFn ? opFoldFn(getOperation()) : failure();
  }
};

struct ConstantOp : public Op<ConstantOp, OpTrait::OneResult,
                              OpTrait::ZeroOperands, OpTrait::ConstantLike> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantOp)
  using Op::Op;
  using FoldAdaptor = FoldAdaptorImpl<ConstantOp>;
  static ArrayRef<StringRef> getAttributeNames() {
    static StringRef names[] = {"value"};
    return names;
  }
  static StringRef getOperationName() { return "fold_test.constant"; }
  OpFoldResult fold(FoldAdaptor) { return getOperation()->getAttr("value"); }
};

struct FoldTestDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldTestDialect)
  static constexpr StringLiteral getDialectNamespace() { return "fold_test"; }
  explicit FoldTestDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<FoldTestDialect>()) {
    addOperations<PartialFoldOp, ConstantOp>();
  }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) final {
    OperationState state(loc, ConstantOp::getOperationName());
    state.addAttribute("value", value);
    state.addTypes(type);
    return builder.create(state);
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
  }

  ~OpFoldResultsTest() override {
    // Destroy users before the ops that define their operands.
    for (Operation *op : llvm::reverse(ops))
      op->destroy();
    opFoldFn = nullptr;
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
  void loadDynamicDialect();

  MLIRContext context;
  Builder builder;
  Type i32;
  Type f32;
  SmallVector<Operation *> ops;
  std::function<OpFoldResults(Operation *)> foldFn;
  std::function<LogicalResult(Operation *, SmallVectorImpl<OpFoldResult> &)>
      legacyFoldFn;
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

TEST_F(OpFoldResultsTest, InitializerList) {
  Operation *producer = createOp({i32});
  Operation *op = createOp({i32, i32});
  Value value = producer->getResult(0);
  Attribute attr = builder.getI32IntegerAttr(1);

  NormalizedOpFoldResults result = normalize(op, {value, attr});
  EXPECT_TRUE(result.replacesAll());
  ASSERT_EQ(result.getReplacements().size(), 2u);
  EXPECT_EQ(result.getReplacements()[0], OpFoldResult(value));
  EXPECT_EQ(result.getReplacements()[1], OpFoldResult(attr));

  // A null element keeps its result.
  NormalizedOpFoldResults partial = normalize(op, {nullptr, attr});
  EXPECT_TRUE(partial.replacesAny());
  EXPECT_FALSE(partial.replacesAll());
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
  Operation *op = createOp({i32, i32}, "fold_test.partial");
  opFoldFn = [&](Operation *) -> OpFoldResults {
    return replacements({producer->getResult(0), OpFoldResult()});
  };
  EXPECT_DEATH((void)op->fold(), "incorrect fold result type");
}

TEST_F(OpFoldResultsDeathTest, ReplacementCountMismatch) {
  Operation *op = createOp({i32, i32}, "fold_test.partial");
  Attribute attr = builder.getI32IntegerAttr(1);
  opFoldFn = [&](Operation *) -> OpFoldResults {
    return replacements({attr, attr, attr});
  };
  EXPECT_DEATH((void)op->fold(),
               "expected one replacement per operation result");
}

#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST

TEST_F(OpFoldResultsTest, DefaultAndSingleReplacement) {
  Operation *op = createOp({i32});
  Attribute attr = builder.getI32IntegerAttr(3);

  EXPECT_TRUE(failed(normalize(op, OpFoldResults())));

  NormalizedOpFoldResults single = normalize(op, OpFoldResult(attr));
  EXPECT_TRUE(single.replacesAll());
  ASSERT_EQ(single.getReplacements().size(), 1u);
  EXPECT_EQ(single.getReplacements()[0], OpFoldResult(attr));

  // A null replacement, or the op's own result, keeps the result.
  EXPECT_TRUE(failed(normalize(op, OpFoldResult())));
  EXPECT_TRUE(failed(normalize(op, OpFoldResult(op->getResult(0)))));
}

TEST_F(OpFoldResultsTest, ProducerInPlaceBit) {
  Operation *op = createOp({i32, i32});
  Attribute attr = builder.getI32IntegerAttr(5);

  OpFoldResults partial = replacements({OpFoldResult(), attr});
  partial.setModifiedInPlace();
  NormalizedOpFoldResults result = normalize(op, std::move(partial));
  EXPECT_TRUE(result.modifiedInPlace());
  EXPECT_TRUE(result.replacesAny());
  EXPECT_FALSE(result.replacesAll());
}

TEST_F(OpFoldResultsTest, ValueReplacement) {
  Operation *producer = createOp({i32});
  Operation *op = createOp({i32});
  Value value = producer->getResult(0);

  NormalizedOpFoldResults result = normalize(op, value);
  EXPECT_TRUE(result.replacesAll());
  EXPECT_EQ(result.getReplacements()[0], OpFoldResult(value));

  EXPECT_TRUE(failed(normalize(op, Value())));
  EXPECT_TRUE(failed(normalize(op, op->getResult(0))));
}

TEST_F(OpFoldResultsTest, IncrementalForm) {
  Operation *op = createOp({i32, i32, i32});
  Attribute first = builder.getI32IntegerAttr(1);
  Attribute second = builder.getI32IntegerAttr(2);

  EXPECT_TRUE(failed(normalize(op, OpFoldResults(op))));

  OpFoldResults results(op);
  results.replace(op->getResult(0), first);
  results.replace(op->getResult(0), second);
  results.replace(op->getResult(1), first);
  // A replacement by the op's own result keeps that result.
  results.replace(op->getResult(1), op->getResult(1));
  results.replace(op->getResult(2), first);
  results.replace(op->getResult(2), OpFoldResult());
  NormalizedOpFoldResults result = normalize(op, std::move(results));
  EXPECT_TRUE(result.replacesAny());
  EXPECT_FALSE(result.replacesAll());
  ASSERT_EQ(result.getReplacements().size(), 3u);
  EXPECT_EQ(result.getReplacements()[0], OpFoldResult(second));
  EXPECT_FALSE(result.getReplacements()[1]);
  EXPECT_FALSE(result.getReplacements()[2]);
}

TEST_F(OpFoldResultsTest, FromLegacy) {
  Operation *op = createOp({i32, i32});
  Attribute attr = builder.getI32IntegerAttr(7);

  EXPECT_TRUE(failed(normalize(op, OpFoldResults::fromLegacy(failure(), {}))));

  NormalizedOpFoldResults inPlace =
      normalize(op, OpFoldResults::fromLegacy(success(), {}));
  EXPECT_TRUE(inPlace.modifiedInPlace());
  EXPECT_FALSE(inPlace.replacesAny());

  NormalizedOpFoldResults all = normalize(
      op, OpFoldResults::fromLegacy(success(), replacements({attr, attr})));
  EXPECT_TRUE(all.replacesAll());
  EXPECT_FALSE(all.modifiedInPlace());
}

//===----------------------------------------------------------------------===//
// Legacy fold APIs
//===----------------------------------------------------------------------===//

// These tests cover the deprecated legacy fold APIs.
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

void OpFoldResultsTest::loadDynamicDialect() {
  context.getOrLoadDynamicDialect("test_fold", [&](DynamicDialect *dialect) {
    auto verify = [](Operation *) { return success(); };
    std::unique_ptr<DynamicOpDefinition> opDef =
        DynamicOpDefinition::get("op", dialect, verify, verify);
    opDef->setFoldHookFn([this](Operation *op, ArrayRef<Attribute>) {
      return NormalizedOpFoldResults(op, foldFn(op));
    });
    dialect->registerDynamicOp(std::move(opDef));

    std::unique_ptr<DynamicOpDefinition> legacyOpDef =
        DynamicOpDefinition::get("legacy_op", dialect, verify, verify);
    legacyOpDef->setFoldHookFn([this](Operation *op, ArrayRef<Attribute>,
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
    noFoldOpDef->setFoldHookFn([this](Operation *op, ArrayRef<Attribute>) {
      return NormalizedOpFoldResults(op, foldFn(op));
    });
    noFoldOpDef->setFoldHookFn(nullptr);
    dialect->registerDynamicOp(std::move(noFoldOpDef));
  });
}

TEST_F(OpFoldResultsTest, UnregisteredOpFoldFails) {
  Operation *op = createOp({i32});
  EXPECT_TRUE(failed(op->fold()));
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
  EXPECT_TRUE(failed(op->fold()));

  foldFn = [](Operation *) -> OpFoldResults { return success(); };
  EXPECT_TRUE(succeeded(op->fold(results)));
  EXPECT_TRUE(results.empty());

  foldFn = [&](Operation *) -> OpFoldResults {
    return replacements({attr, producer->getResult(1)});
  };
  EXPECT_TRUE(succeeded(op->fold(results)));
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0], OpFoldResult(attr));
  EXPECT_EQ(results[1], OpFoldResult(producer->getResult(1)));
  results.clear();

  // The legacy overloads do not apply a partial fold. Without an in-place
  // change, the partial fold is a failure.
  foldFn = [&](Operation *foldedOp) {
    OpFoldResults partial = replacements({OpFoldResult(), attr});
    return partial;
  };
  EXPECT_TRUE(failed(op->fold(results)));
  EXPECT_TRUE(results.empty());
  NormalizedOpFoldResults partialResult = op->fold();
  EXPECT_TRUE(succeeded(partialResult));
  EXPECT_FALSE(partialResult.modifiedInPlace());
  ASSERT_EQ(partialResult.getReplacements().size(), 2u);
  EXPECT_FALSE(partialResult.getReplacements()[0]);
  EXPECT_EQ(partialResult.getReplacements()[1], OpFoldResult(attr));

  // With an in-place change, the partial fold is reported as in place.
  foldFn = [&](Operation *foldedOp) {
    OpFoldResults partial = replacements({OpFoldResult(), attr});
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
  EXPECT_TRUE(failed(op->fold()));
}

TEST_F(OpFoldResultsTest, LegacyDynamicFoldHookKeepsStrictContract) {
  loadDynamicDialect();
  Operation *producer = createOp({i32, i32});
  Operation *op = createOp({i32, i32}, "test_fold.legacy_op");
  Attribute attr = builder.getI32IntegerAttr(1);

  legacyFoldFn = [](Operation *, SmallVectorImpl<OpFoldResult> &) {
    return failure();
  };
  EXPECT_TRUE(failed(op->fold()));

  legacyFoldFn = [](Operation *, SmallVectorImpl<OpFoldResult> &) {
    return success();
  };
  NormalizedOpFoldResults inPlace = op->fold();
  EXPECT_TRUE(succeeded(inPlace));
  EXPECT_TRUE(inPlace.modifiedInPlace());
  EXPECT_FALSE(inPlace.replacesAny());

  legacyFoldFn = [&](Operation *, SmallVectorImpl<OpFoldResult> &results) {
    results.push_back(attr);
    results.push_back(producer->getResult(0));
    return success();
  };
  NormalizedOpFoldResults all = op->fold();
  EXPECT_TRUE(succeeded(all));
  EXPECT_FALSE(all.modifiedInPlace());
  EXPECT_TRUE(all.replacesAll());
  ASSERT_EQ(all.getReplacements().size(), 2u);
  EXPECT_EQ(all.getReplacements()[0], OpFoldResult(attr));
  EXPECT_EQ(all.getReplacements()[1], OpFoldResult(producer->getResult(0)));

  // A legacy fold that forwards the op's own results keeps every result, so
  // it is a failure.
  legacyFoldFn = [](Operation *foldedOp,
                    SmallVectorImpl<OpFoldResult> &results) {
    llvm::append_range(results, foldedOp->getResults());
    return success();
  };
  EXPECT_TRUE(failed(op->fold()));
  SmallVector<OpFoldResult> results;
  EXPECT_TRUE(failed(op->fold(results)));
  EXPECT_TRUE(results.empty());
}

TEST_F(OpFoldResultsTest, LegacyDynamicFoldHookMayForwardAnotherResult) {
  loadDynamicDialect();
  Operation *op = createOp({i32, i32}, "test_fold.legacy_op");
  Attribute attr = builder.getI32IntegerAttr(1);

  // Replacement 0 is result 1, which the fold also replaces. The
  // replaced-result check applies only to `OpFoldResults fold(FoldAdaptor)`.
  legacyFoldFn = [&](Operation *foldedOp,
                     SmallVectorImpl<OpFoldResult> &results) {
    results.push_back(foldedOp->getResult(1));
    results.push_back(attr);
    return success();
  };
  NormalizedOpFoldResults result = op->fold();
  EXPECT_TRUE(result.replacesAll());
  ASSERT_EQ(result.getReplacements().size(), 2u);
  EXPECT_EQ(result.getReplacements()[0], OpFoldResult(op->getResult(1)));
  EXPECT_EQ(result.getReplacements()[1], OpFoldResult(attr));

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
  NormalizedOpFoldResults result = op->fold();
  EXPECT_TRUE(result.replacesAll());
  ASSERT_EQ(result.getReplacements().size(), 1u);
  EXPECT_EQ(result.getReplacements()[0], OpFoldResult(attr));

  legacyFoldFn = [](Operation *, SmallVectorImpl<OpFoldResult> &) {
    return failure();
  };
  EXPECT_TRUE(failed(op->fold()));
}

TEST_F(OpFoldResultsTest, NullFoldHookFails) {
  loadDynamicDialect();
  Operation *op = createOp({i32}, "test_fold.no_fold_op");
  // The removed hook would report an in-place fold.
  foldFn = [](Operation *) -> OpFoldResults { return success(); };
  EXPECT_TRUE(failed(op->fold()));
  SmallVector<OpFoldResult> results;
  EXPECT_TRUE(failed(op->fold(results)));
  EXPECT_TRUE(results.empty());
}

TEST_F(OpFoldResultsTest, OpBuilderLegacyTryFold) {
  Block block;
  OpBuilder b(&context);
  b.setInsertionPointToEnd(&block);
  Location loc = b.getUnknownLoc();
  Value producer =
      b.create(loc, b.getStringAttr("foo.producer"), {}, {i32})->getResult(0);
  Operation *op =
      b.create(loc, b.getStringAttr("fold_test.partial"), {}, {i32, i32});
  Attribute attr = b.getI32IntegerAttr(1);
  SmallVector<Value> results;
  SmallVector<Operation *> constants;

  // A partial fold counts only for its in-place change, and no fold follows
  // it.
  unsigned calls = 0;
  bool inPlace = false;
  opFoldFn = [&](Operation *) {
    ++calls;
    OpFoldResults result{attr, nullptr};
    result.setModifiedInPlace(inPlace);
    return result;
  };
  EXPECT_TRUE(failed(b.tryFold(op, results, &constants)));
  inPlace = true;
  EXPECT_TRUE(succeeded(b.tryFold(op, results, &constants)));
  EXPECT_EQ(calls, 2u);
  EXPECT_TRUE(results.empty());
  EXPECT_TRUE(constants.empty());

  // A full fold materializes its constants.
  opFoldFn = [&](Operation *) -> OpFoldResults { return {attr, producer}; };
  ASSERT_TRUE(succeeded(b.tryFold(op, results, &constants)));
  ASSERT_EQ(results.size(), 2u);
  ASSERT_EQ(constants.size(), 1u);
  EXPECT_EQ(results[0], constants[0]->getResult(0));
  EXPECT_EQ(results[1], producer);
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
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

LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
