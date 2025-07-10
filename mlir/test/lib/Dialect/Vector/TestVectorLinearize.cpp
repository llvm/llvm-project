//===- TestVectorLinearize.cpp - Test Vector linearization ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math//IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;

namespace {

struct TestVectorLinearize final
    : public PassWrapper<TestVectorLinearize, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorLinearize)

  StringRef getArgument() const override { return "test-vector-linearize"; }
  StringRef getDescription() const override {
    return "Use shape_casts to ensure vector operands/results are rank <= 1";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect, arith::ArithDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    Operation *op = getOperation();

    // Step 1: Run the linearization patterns.
    //
    // Note that we disable folding to prevent the extract(shape_cast) ->
    // extract folder undoing linearization. Without disabling this, we can get
    // into infinite loops.
    {
      RewritePatternSet patterns(&context);
      populateForVectorLinearize(patterns);
      GreedyRewriteConfig config;
      config.enableFolding(false);
      if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
        return signalPassFailure();
    }

    // Step 2: linearize SCF structured ops using type conversion.
    {
      TypeConverter typeConverter;
      RewritePatternSet patterns(&context);
      ConversionTarget target(context);

      // Convert 'type' to a "legal" (rank-1) type.
      auto convertType = [](Type type) -> std::optional<Type> {
        VectorType vectorType = dyn_cast<VectorType>(type);
        if (!vectorType || !isLinearizableVector(vectorType))
          return type;

        VectorType linearizedType = VectorType::get(vectorType.getNumElements(),
                                                    vectorType.getElementType(),
                                                    vectorType.isScalable());
        return linearizedType;
      };
      typeConverter.addConversion(convertType);

      // This function is used during legalization to create shape_casts between
      // the legal rank-1 types and other types.
      auto materializeCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
        if (inputs.size() != 1)
          return nullptr;

        Value input = inputs.front();
        if (!isa<VectorType>(type) || !isa<VectorType>(input.getType()))
          return nullptr;

        return builder.create<vector::ShapeCastOp>(loc, type, input);
      };
      typeConverter.addSourceMaterialization(materializeCast);
      typeConverter.addTargetMaterialization(materializeCast);

      // As we are here just illustrating how to use type conversion to
      // linearize SCF operations, we consider all other operations already
      // legal.
      target.markUnknownOpDynamicallyLegal(
          [=](Operation *op) -> std::optional<bool> {
            if (scf::SCFDialect::getDialectNamespace() !=
                op->getDialect()->getNamespace())
              return true;

            // This will return true if, for all operand and result types `t`,
            // convertType(t) = t. This is true if there are no rank>=2 vectors.
            return typeConverter.isLegal(op);
          });

      mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
          typeConverter, patterns, target);
      if (failed(applyPartialConversion(op, target, std::move(patterns))))
        return signalPassFailure();
    }

    // Step 3: Perform folding.
    if (failed(applyPatternsGreedily(op, RewritePatternSet(&context))))
      return signalPassFailure();
  }
};

struct TestRankReduceStridedSliceOps final
    : public PassWrapper<TestRankReduceStridedSliceOps, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRankReduceStridedSliceOps)

  TestRankReduceStridedSliceOps() = default;
  TestRankReduceStridedSliceOps(const TestRankReduceStridedSliceOps &pass) =
      default;

  StringRef getArgument() const override {
    return "test-rank-reduce-strided-slice-ops";
  }
  StringRef getDescription() const override {
    return "Test pass for rank-reducing strided slice ops.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateForStridedRankReduction(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

struct TestVectorBitWidthLinearize final
    : public PassWrapper<TestVectorBitWidthLinearize, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorBitWidthLinearize)

  TestVectorBitWidthLinearize() = default;
  TestVectorBitWidthLinearize(const TestVectorBitWidthLinearize &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const override {
    return "test-bit-width-constrained-vector-linearize";
  }
  StringRef getDescription() const override {
    return "Linearizes ND vectors for N >= 2 into 1D vectors, with constraints "
           "on inner-most dimension's bit width. If the inner-most dimension "
           "exceded a threshold, the op is not linearized.";
  }
  Option<unsigned> targetVectorBitwidth{
      *this, "target-vector-bitwidth",
      llvm::cl::desc(
          "Minimum vector bitwidth to enable the flattening transformation"),
      llvm::cl::init(std::numeric_limits<unsigned>::max())};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    Operation *op = getOperation();

    // Initialize the patterns with a pre-condition on the the bit-width, for
    // linearization.
    auto preCondition = [&](Operation *op) -> LogicalResult {
      bool notLinearizable =
          isNotLinearizableBecauseLargeInnerDimension(op, targetVectorBitwidth);
      return notLinearizable ? failure() : success();
    };
    RewritePatternSet patterns(&context);
    populateForVectorLinearize(patterns, preCondition);

    // Apply the patterns, with folding disabled.
    if (failed(
            applyPatternsGreedily(op, std::move(patterns),
                                  GreedyRewriteConfig().enableFolding(false))))
      return signalPassFailure();

    // Fold.
    if (failed(applyPatternsGreedily(op, RewritePatternSet(&context))))
      return signalPassFailure();
  }

  /// If `type` is VectorType with trailing dimension of (bit) size greater than
  /// or equal to `targetBitWidth`, its defining op is considered legal.
  static bool
  isNotLinearizableBecauseLargeInnerDimension(Type type,
                                              unsigned targetBitWidth) {

    VectorType vecType = dyn_cast<VectorType>(type);

    // Not linearizable for reasons other than what this function checks.
    if (!vecType || vecType.getRank() == 0)
      return false;

    // The width of the type 'index' is unbounded (and therefore potentially
    // above the target width).
    if (vecType.getElementType().isIndex())
      return true;

    unsigned finalDimSize = vecType.getShape().back();
    unsigned nbBitsPerElm = vecType.getElementTypeBitWidth();
    unsigned trailingVecDimBitWidth = finalDimSize * nbBitsPerElm;
    return trailingVecDimBitWidth >= targetBitWidth;
  }

private:
  static bool
  isNotLinearizableBecauseLargeInnerDimension(Operation *op,
                                              unsigned targetBitWidth) {
    // Check on bitwidths.
    SmallVector<std::pair<Type, unsigned>> toCheck =
        getTypeBitWidthBoundPairs(op, targetBitWidth);
    return std::any_of(toCheck.begin(), toCheck.end(),
                       [&](std::pair<Type, unsigned> typeWidth) {
                         return isNotLinearizableBecauseLargeInnerDimension(
                             typeWidth.first, typeWidth.second);
                       });
  }

  /// Get the set of operand/result types to check for sufficiently
  /// small inner-most dimension size.
  static SmallVector<std::pair<Type, unsigned>>
  getTypeBitWidthBoundPairs(Operation *op, unsigned targetBitWidth) {

    if (auto insertOp = dyn_cast<InsertOp>(op)) {
      unsigned w = targetBitWidth < std::numeric_limits<unsigned>::max()
                       ? targetBitWidth + 1
                       : targetBitWidth;
      return {{insertOp.getValueToStoreType(), w}};
    }

    auto resultTypes = op->getResultTypes();
    SmallVector<std::pair<Type, unsigned>> resultsWithBitWidth;
    resultsWithBitWidth.reserve(resultTypes.size());
    for (Type type : resultTypes) {
      resultsWithBitWidth.push_back({type, targetBitWidth});
    }
    return resultsWithBitWidth;
  }
};

} // namespace

namespace mlir {
namespace test {
extern void registerTestVectorLinearize() {
  PassRegistration<TestVectorLinearize>();
  PassRegistration<TestVectorBitWidthLinearize>();
  PassRegistration<TestRankReduceStridedSliceOps>();
}
} // namespace test
} // namespace mlir
